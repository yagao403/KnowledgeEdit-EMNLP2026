from contextlib import nullcontext
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, Qwen3ForCausalLM
from peft import PeftModel, PeftMixedModel
from typing import Literal

from training import IGNORE_INDEX

def get_n_tokens_per_sample(batch) -> torch.Tensor:
    student_masks = batch['student_masks'][..., 1:]  # (batch_size, seq_length)
    n_tokens_per_sample = student_masks.sum(-1).to(student_masks.device)  # (batch_size,)
    return n_tokens_per_sample

def kl(
    student_logits: torch.Tensor,   # (n_tokens, vocab_size)
    ref_logits: torch.Tensor,       # (n_tokens, vocab_size)
    temperature: float,
    compute_reverse_kl: bool = True,
    compute_forward_kl: bool = False,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    log_p_s = F.log_softmax(student_logits / temperature, dim=-1)
    log_p_ref = F.log_softmax(ref_logits / temperature, dim=-1)

    rkl = fkl = None
    if compute_forward_kl:
        p_s = log_p_s.exp()
        fkl = (p_s * (log_p_s - log_p_ref)).sum(dim=-1)  # (n_tokens,)

    if compute_reverse_kl:
        p_ref = log_p_ref.exp()
        rkl = (p_ref * (log_p_ref - log_p_s)).sum(dim=-1)  # (n_tokens,)

    return rkl, fkl  # each: (n_tokens,) or None


def reverse_kl(
    student_logits: torch.Tensor,   # (n_tokens, vocab_size)
    teacher_logits: torch.Tensor,    # (n_tokens, vocab_size)
    temperature: float,
) -> torch.Tensor:
    """Reverse KL loss used for distillation."""
    s_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    t_log_probs = F.log_softmax(teacher_logits / temperature, dim=-1)

    return F.kl_div(
        s_log_probs, t_log_probs, log_target=True,
        reduction="none"
    ).sum(dim=-1)  # (n_tokens,)


def reduce_to_sample(
    loss_t: torch.Tensor | None,
    student_masks: torch.Tensor
) -> torch.Tensor | None:
    if loss_t is None:
        return None
    batch_size, seq_length = student_masks.shape
    loss_mx = torch.zeros(batch_size, seq_length, device=student_masks.device, dtype=loss_t.dtype)
    loss_mx[student_masks] = loss_t
    return loss_mx.sum(-1) / student_masks.sum(-1)  # (batch_size,)

def reduce_to_batch(
    sampled_loss: torch.Tensor,
    student_masks: torch.Tensor
) -> torch.Tensor:
    assert sampled_loss is not None
    seq_lengths = student_masks.sum(-1)  # (batch_size,)
    total_tokens = seq_lengths.sum()
    total_loss = (sampled_loss * seq_lengths).sum()
    return total_loss / total_tokens  # scalar

def apply_weights_and_reduce(
    kl: torch.Tensor | None,  # (n_tokens,) or None
    token_weights: torch.Tensor | None,
    reduction: Literal["batch", "sample"],
    student_masks: torch.Tensor  # (batch_size, seq_length)
):
    assert reduction in ["batch", "sample"]
    if kl is not None:
        if token_weights is not None:
            kl = kl * token_weights
        if reduction == "batch":
            kl = kl.sum() / student_masks.sum()  # divide by total number of tokens in the batch - shape: scalar
    return kl  # (n_tokens,) or scalar or None

def compute_kl(
    batch,
    student: PeftModel | PeftMixedModel,
    temperature,
    reduction: Literal["batch", "sample"] = "sample",  # "batch" is not currently used anywhere
    eval: bool = False,
    compute_forward_kl: bool = False,
    compute_reverse_kl: bool = True,
    compute_ce: bool = False,
    teacher: AutoModelForCausalLM | PeftModel | PeftMixedModel | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """
    teacher: if None, use the student's base as teacher (i.e., "student_base" distillation)

    Returns:
    - rkl | None: Reverse KL divergence (teacher || student)
    - fkl | None: Forward KL divergence (student || teacher)
    - ce | None: Cross-entropy loss of the student
    """
    assert reduction in ["batch", "sample"]

    student_inputs = batch['student_seqs'][..., :-1]  # (batch_size, seq_length)
    student_masks = batch['student_masks'][..., 1:]  # (batch_size, seq_length)

    # Calculate token level loss weights
    loss_weights = batch.get('loss_weights', None)  # (batch_size,)

    if loss_weights is not None:
        token_weights = loss_weights.unsqueeze(-1).expand_as(student_masks)[student_masks]  # (n_tokens,)
    else:
        token_weights = None

    active_adapter = student.active_adapter
    if teacher is None:
        teacher = student
        teacher_context = student.disable_adapter()
    else:
        teacher_context = nullcontext()
    with torch.no_grad(), teacher_context:
        teacher.eval()
        teacher_inputs = batch['teacher_seqs'][..., :-1]
        teacher_masks = batch['teacher_masks'][..., 1:]
        teacher_output = teacher.forward(teacher_inputs)
        teacher_logits = teacher_output.logits[teacher_masks].clone()
        del teacher_output
        torch.cuda.empty_cache()
    student.set_adapter(active_adapter)

    if eval:
        student.eval()
    else:
        student.train()  # This is important! If we use "student_base" as a teacher, doing teacher.eval() seems
                     # to break things. For example, gradient checkpointing did not work without this line.
    student_output = student.forward(student_inputs)
    student_logits = student_output.logits[student_masks]
    rkl = fkl = None
    if compute_reverse_kl and not compute_forward_kl:
        # Use torch's built-in kl_div for reverse KL
        rkl = reverse_kl(student_logits, teacher_logits, temperature)  # (n_tokens, )
    else:
        rkl, fkl = kl(
            student_logits, teacher_logits, temperature,
            compute_reverse_kl=compute_reverse_kl,
            compute_forward_kl=compute_forward_kl
        )  # each: (n_tokens,) or None

    rkl = apply_weights_and_reduce(rkl, token_weights, reduction, student_masks)
    fkl = apply_weights_and_reduce(fkl, token_weights, reduction, student_masks)

    ce = None
    if compute_ce:
        # Token loss is only used for evaluation, for training use compute_token_loss()
        student_labels = batch["student_seqs"][batch["student_masks"]]  # (n_tokens,)
        ce = F.cross_entropy(student_logits, student_labels,
            ignore_index=IGNORE_INDEX, reduction="none")
        # We do not apply the weights because the CE loss is only used for evaluation
        if reduction == "batch":
            ce = ce.sum() / student_masks.sum()  # divide by total number of tokens in the batch - shape: scalar

    if reduction == "sample":
        rkl = reduce_to_sample(rkl, student_masks)
        fkl = reduce_to_sample(fkl, student_masks)
        ce = reduce_to_sample(ce, student_masks)

    return rkl, fkl, ce


def get_attention_mask(target_mask: torch.Tensor) -> torch.Tensor:
    x = target_mask

    # 1. flip so the last 1 becomes the first 1 in each row
    y = torch.flip(x, dims=[1]).to(torch.int)

    # 2. cumulative sum → positions at or after that first-from-right 1 become ≥1
    y_filled = (y.cumsum(dim=1) > 0)

    # 3. flip back to original order
    mask = torch.flip(y_filled, dims=[1])
    return mask


def compute_ce_loss(
    batch,
    student: PeftModel | PeftMixedModel,
    reduction: Literal["batch", "sample"] = "sample",
    eval: bool = False
) -> torch.Tensor:
    assert reduction in ["batch", "sample"]

    student_inputs = batch['student_seqs'][..., :-1]  # (batch_size, seq_length)
    student_masks = batch['student_masks'][..., 1:]  # (batch_size, seq_length)

    if eval:
        student.eval()
    else:
        student.train()  # This is important! If we use "student_base" as a teacher, doing teacher.eval() seems
                     # to break things. For example, gradient checkpointing did not work without this line.

    student_output = student.forward(student_inputs)
    student_logits = student_output.logits[student_masks]
    student_labels = batch["student_seqs"][batch["student_masks"]]
    ce = F.cross_entropy(
        student_logits,
        student_labels,
        ignore_index=IGNORE_INDEX,
        reduction="mean" if reduction == "batch" else "none"
    )  # if reduction = "sample": (n_tokens,)

    if reduction == "sample":
        ce = reduce_to_sample(ce, student_masks)

    return ce
