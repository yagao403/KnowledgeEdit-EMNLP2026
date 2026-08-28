"""Launch the paper's single-hop supervised fine-tuning baseline."""

from __future__ import annotations

import argparse

import torch.multiprocessing as mp

from configs.train_distillation import _expand, _model
from core import LOGBOOK_PATH
from core.llm import LLM
from core.tokenizer import Tokenizer
from training.dataloaders import InfDataLoader, ValidationDataLoader
from training.train_sft_distil import MyHyperparameters, train_sft
from training.utils import ComputeConfig, ModelConfig, RunConfig, TrainingData, ValidationData


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="append", required=True, help="XML path or glob; repeat to combine sources")
    parser.add_argument("--validation", action="append", default=[], help="Validation XML path or glob")
    parser.add_argument("--dataset", choices=("fictbio", "mquake-cf", "recoe"), required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--base-model", default="qwen3-32b")
    parser.add_argument("--student-adapter")
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--max-teacher-seq-len", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=30.0)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--lora-rank", type=int, default=128)
    parser.add_argument(
        "--lora-a-weight-decay-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply weight decay only to LoRA-A parameters (enabled in the paper runs)",
    )
    parser.add_argument("--val-interval", type=int, default=160)
    parser.add_argument("--checkpoint-interval", type=int, default=120)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--wandb-project", default="edit-knowledge-not-just-facts")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train_paths = _expand(args.train)
    validation_paths = _expand(args.validation)
    if not train_paths:
        raise FileNotFoundError(f"No training XML files matched: {args.train}")
    if args.validation and not validation_paths:
        raise FileNotFoundError(f"No validation XML files matched: {args.validation}")
    print(f"train={len(train_paths)} validation={len(validation_paths)} epochs={args.epochs}")
    if args.dry_run:
        return

    base_model = _model(args.base_model)
    tokenizer = Tokenizer(str(base_model.tokenizer_id))
    train_data = TrainingData.with_single_group(
        train_paths,
        micro_batch_size=args.micro_batch_size,
        n_micro_batches_in_batch=args.gradient_accumulation,
        max_teacher_seq_len=args.max_teacher_seq_len,
        student_dropout_rate=0.0,
    )
    train_loader = InfDataLoader.from_training_data(train_data, only_student=True, tokenizer=tokenizer)
    validation_loader = None
    if validation_paths:
        validation_data = ValidationData.with_single_group(validation_paths, args.micro_batch_size)
        validation_loader = ValidationDataLoader.from_validation_data(
            validation_data,
            only_student=True,
            tokenizer=tokenizer,
        )

    compute = ComputeConfig(
        world_size=args.world_size,
        tp_size=args.tp_size,
        gradient_checkpointing=True,
        use_fsdp_only=args.tp_size == 1,
        torch_dtype="bf16",
    )
    model = ModelConfig(
        base_model=base_model,
        student=LLM.from_adapter(args.student_adapter) if args.student_adapter else None,
        teacher_type="student_base",
        lora_target_modules="full",
        lora_r=args.lora_rank,
    )
    project_path = LOGBOOK_PATH / "training" / "knowledge-editing-sft"
    project_path.mkdir(parents=True, exist_ok=True)
    run = RunConfig(
        project_name="edit-knowledge-not-just-facts",
        run_name=args.run_name,
        project_path=project_path,
        wnb_project=args.wandb_project,
        group_name=f"{args.dataset}-sft",
        model_cfg=model,
        compute_cfg=compute,
        val_interval=args.val_interval,
        log_interval=args.log_interval,
        checkpoint_interval=args.checkpoint_interval,
        latest_checkpoint_interval=args.checkpoint_interval,
        use_wandb=args.wandb,
        notes="Single-hop SFT baseline",
    )
    hyperparameters = MyHyperparameters(
        max_lr=args.learning_rate,
        weight_decay=args.weight_decay,
        n_epochs=args.epochs,
        warmup_steps=0,
        temperature=args.temperature,
        lora_a_weight_decay_only=args.lora_a_weight_decay_only,
        w_sft=1.0,
        w_distil=0.0,
        w_reg=0.0,
    )
    mp.spawn(
        train_sft,
        args=(train_loader, validation_loader, None, None, None, None, run, hyperparameters),
        nprocs=compute.world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
