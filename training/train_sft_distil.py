import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import contextlib
from dataclasses import dataclass
import torch
from typing import Literal

from core.llm import LLM
from training.metrics import Aggregator
from training.losses import compute_kl, compute_ce_loss, get_n_tokens_per_sample, reduce_to_batch
from training.utils import (
    Run, validate, Hyperparameters,
    RunConfig, create_student, load_student, to_device, create_optimizer,
    set_max_steps_from_n_epochs, create_tagged_metrics
)
from training.dataloaders import InfDataLoader, ValidationDataLoader

@dataclass(kw_only=True)
class MyHyperparameters(Hyperparameters):
    w_sft: float  # The weight for the cross-entropy loss
    w_distil: float  # The weight for the distillation KL loss
    w_reg: float  # The weight for the forward KL loss (regularization)
    temperature: float = 1.0  # Temperature for the distillation KL loss
    reg_loss: Literal["rkl", "fkl", "jsd"] = "rkl"
    fkl_temperature: float = 1.0  # Temperature for the forward KL loss (regularization)


def train_sft(
    rank,
    train_sft_loader: InfDataLoader,
    val_sft_loader: ValidationDataLoader | None,  # Cross-entropy validation loader
    train_reg_loader: InfDataLoader | None,  # Forward KL data (regularization)
    val_reg_loader: ValidationDataLoader | None,  # Forward KL validation loader (regularization)
    train_distil_loader: InfDataLoader | None,  # Distillation (reverse KL) data
    val_distil_loader: ValidationDataLoader | None,  # Distillation (reverse KL) validation loader
    run_cfg: RunConfig,
    lp: MyHyperparameters,
) -> None:
    assert run_cfg.val_interval % run_cfg.log_interval == 0, "eval_interval must be divisible by log_interval"

    run = Run(run_cfg, lp)
    run.setup(rank)
    # Build config for logging using loader-provided to_dict() helpers
    config = {
        "hyperparameters": lp.to_dict(),
        "train_sft_data": train_sft_loader.to_dict(),
        "val_sft_data": val_sft_loader.to_dict() if (val_sft_loader is not None) else None,
        "train_distil_data": train_distil_loader.to_dict() if (train_distil_loader is not None) else None,
        "val_distil_data": val_distil_loader.to_dict() if (val_distil_loader is not None) else None,
        "train_reg_data": train_reg_loader.to_dict() if (train_reg_loader is not None) else None,
        "val_reg_data": val_reg_loader.to_dict() if (val_reg_loader is not None) else None,
    }
    run.setup_wandb_new(config)

    if run.run_cfg.model_cfg.student is None:
        base_llm = run.run_cfg.model_cfg.base_model
        student, model_config, peft_config = create_student(
            rank,
            device_mesh=run.device_mesh,
            use_ema=getattr(lp, 'ema_alpha', None) is not None,
            model_cfg=run.run_cfg.model_cfg,
            compute_cfg=run.run_cfg.compute_cfg,
        )
    else:
        assert run.run_cfg.model_cfg.student is not None
        student_llm = run.run_cfg.model_cfg.student
        base_llm = LLM(student_llm.model_id, [], str(student_llm.tokenizer_id))
        student, model_config, peft_config = load_student(
            rank,
            device_mesh=run.device_mesh,
            use_ema=getattr(lp, 'ema_alpha', None) is not None,
            model_cfg=run.run_cfg.model_cfg,
            compute_cfg=run.run_cfg.compute_cfg,
        )

    run.print(f"Student model: {student}")

    student.train()
    optimizer = create_optimizer(
        student,
        lora_a_weight_decay_only=run.hp.lora_a_weight_decay_only,
        lora_b_weight_decay_only=run.hp.lora_b_weight_decay_only,
        lr=run.hp.max_lr,
        weight_decay=run.hp.weight_decay,
    )
    run.print_gpu_utilization()

    # Initialize loaders inside the worker process and create iterators
    assert isinstance(run.run_cfg.compute_cfg.dp_size, int)
    len_sft_trainset = train_sft_loader.reset(run.dp_rank, run.run_cfg.compute_cfg.dp_size)
    iter_sft_dataloader = iter(train_sft_loader)

    if train_distil_loader is not None:
        train_distil_loader.reset(run.dp_rank, run.run_cfg.compute_cfg.dp_size)
        iter_distil_dataloader = iter(train_distil_loader)
    else:
        iter_distil_dataloader = None

    if train_reg_loader is not None:
        train_reg_loader.reset(run.dp_rank, run.run_cfg.compute_cfg.dp_size)
        iter_reg_dataloader = iter(train_reg_loader)
    else:
        iter_reg_dataloader = None

    # Compute max steps based on SFT loader config
    lp.max_steps = max_steps = lp.max_steps or set_max_steps_from_n_epochs(
        len_sft_trainset,
        micro_batch_size=train_sft_loader.micro_batch_size,
        n_micro_batches_per_batch=train_sft_loader.n_micro_batches_in_batch,
        n_devices=run.devices,
        n_epochs=lp.n_epochs,
        verbose=run.is_main_process,
    )

    # Validation data
    if val_sft_loader is not None:
        val_sft_loader.reset(run.dp_rank, run.run_cfg.compute_cfg.dp_size)

    if val_distil_loader is not None:
        val_distil_loader.reset(run.dp_rank, run.run_cfg.compute_cfg.dp_size)

    if val_reg_loader is not None:
        val_reg_loader.reset(run.dp_rank, run.run_cfg.compute_cfg.dp_size)


    batch_size = train_sft_loader.n_micro_batches_in_batch * train_sft_loader.micro_batch_size * run.devices
    run.print(f"Number of steps in one epoch: {len_sft_trainset // batch_size}")

    train_metrics = {}
    for step in range(max_steps):
        run.reset_step(step, max_steps)

        # Validation
        if step % run_cfg.val_interval == 0:
            if val_sft_loader is not None:
                validate(
                    student, val_sft_loader, rank, run,  # type: ignore
                    compute_ce=True, compute_forward_kl=False, compute_reverse_kl=False,
                    name_prefix="val/sft")
            if val_distil_loader is not None:
                validate(
                    student, val_distil_loader, rank, run,  # type: ignore
                    compute_ce=False, compute_forward_kl=False, compute_reverse_kl=True,
                    name_prefix="val/distil")
            if val_reg_loader is not None:
                validate(
                    student, val_reg_loader, rank, run,  # type: ignore
                    compute_ce=False, compute_forward_kl=True, compute_reverse_kl=True,
                    name_prefix="val/reg")
            if step > 0:
                run.export_adapters(
                    rank=run.global_rank, model=student, base_llm=base_llm,
                    hf_model_config=model_config, lora_config=peft_config,
                    device_mesh=run.device_mesh)

        run.reset_step_time()
        lr = lp.adjust_learning_rate(optimizer, step)
        run.add_to_metrics("train/lr", lr)

        student.train()
        # Supervised fine-tuning with cross-entropy loss
        acc_sft = 0
        aggregator = Aggregator(
            rank,
            dp_group=run.device_mesh.get_group(mesh_dim="dp"),
            dp_rank=run.dp_rank,
            tp_rank=run.tp_rank,
        )
        for _ in range(train_sft_loader.n_micro_batches_in_batch):
            batch = next(iter_sft_dataloader)
            batch = to_device(batch, rank)
            run.print_gpu_utilization()

            # context = student.no_sync() if accum_step_id < hp.n_logit_micro_batches_per_batch-1 else contextlib.nullcontext()
            context = contextlib.nullcontext() # Currently disable no_sync
            with context:
                loss_micro = compute_ce_loss(batch, student, reduction="sample")  # scalar

                n_tokens_per_sample = get_n_tokens_per_sample(batch)
                tagged_metrics = create_tagged_metrics(
                    batch["tags"], batch["drop"], loss_micro, n_tokens_per_sample, "train/sft/loss")

                aggregator.add_batch(tagged_metrics)
                loss_micro = reduce_to_batch(loss_micro, batch["student_masks"])
                loss = lp.w_sft * loss_micro / train_sft_loader.n_micro_batches_in_batch
                acc_sft += loss.item()
                loss.backward()

        if (train_distil_loader is not None) and (iter_distil_dataloader is not None):
            # Distillation with reverse KL
            acc_distil = 0
            for _ in range(train_distil_loader.n_micro_batches_in_batch):
                batch = next(iter_distil_dataloader)  # type: ignore
                batch = to_device(batch, rank)
                run.print_gpu_utilization()
                T = lp.temperature
                # TODO: , reduction="batch"
                loss_micro, _, _ = compute_kl(batch, student, temperature=T, reduction="sample")
                assert loss_micro is not None

                n_tokens_per_sample = get_n_tokens_per_sample(batch)
                tagged_metrics = create_tagged_metrics(
                    batch["tags"], batch["drop"], loss_micro, n_tokens_per_sample, "train/distil/loss")

                aggregator.add_batch(tagged_metrics)
                loss_micro = reduce_to_batch(loss_micro, batch["student_masks"])

                # Scale the loss by T^2 as in the knowledge distillation paper
                loss = lp.w_distil * T * T * loss_micro / train_distil_loader.n_micro_batches_in_batch
                acc_distil += loss.item()
                loss.backward()

        # Regularization
        if (train_reg_loader is not None) and (iter_reg_dataloader is not None):
            acc = 0
            for _ in range(train_reg_loader.n_micro_batches_in_batch):
                batch = next(iter_reg_dataloader)  # type: ignore
                batch = to_device(batch, rank)
                run.print_gpu_utilization()
                # TODO: , reduction="batch"
                rkl, fkl, _ = compute_kl(
                    batch, student, temperature=lp.temperature,
                    compute_forward_kl=lp.reg_loss in ("fkl", "jsd"),
                    compute_reverse_kl=lp.reg_loss in ("rkl", "jsd"),
                    reduction="sample",
                )
                if lp.reg_loss == "rkl":
                    loss_micro = rkl
                elif lp.reg_loss == "fkl":
                    loss_micro = fkl
                else:  # both
                    loss_micro = 0.5 * rkl + 0.5 * fkl  # type: ignore
                assert loss_micro is not None

                n_tokens_per_sample = get_n_tokens_per_sample(batch)
                tagged_metrics = create_tagged_metrics(
                    batch["tags"], batch["drop"], loss_micro, n_tokens_per_sample, "train/reg/loss")

                aggregator.add_batch(tagged_metrics)
                loss_micro = reduce_to_batch(loss_micro, batch["student_masks"])

                loss = lp.w_reg * loss_micro / train_reg_loader.n_micro_batches_in_batch
                acc += loss.item()
                loss.backward()

        logit_metrics_total, logit_metrics_by_group = aggregator.get_average()
        train_metrics.update(logit_metrics_total)
        train_metrics.update(logit_metrics_by_group)
        run.add_dict_to_metrics(train_metrics)

        if lp.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(student.parameters(), lp.max_grad_norm)

        optimizer.step()
        optimizer.zero_grad()

        # Release CUDA cache to reduce memory fragmentation
        torch.cuda.empty_cache()
        run.log_step()

        if run.is_checkpoint_step():
            run.export_adapters(
                rank=run.global_rank, model=student, base_llm=base_llm,
                hf_model_config=model_config, lora_config=peft_config,
                device_mesh=run.device_mesh)
            run.save_optimizer_checkpoint(optimizer)

    run.export_adapters(
        rank=run.global_rank, model=student, base_llm=base_llm,
        hf_model_config=model_config, lora_config=peft_config,
        device_mesh=run.device_mesh)
    run.save_optimizer_checkpoint(optimizer)

    run.print("Done")

