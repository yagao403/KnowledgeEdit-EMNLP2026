from dataclasses import dataclass
import torch
from pathlib import Path
from typing import Callable

from core.llm import LLM
from core.tokenizer import Tokenizer
from core.steps import STStepMessages
from training.losses import compute_kl, reduce_to_batch
from training.utils import (
    Run,
    validate, Hyperparameters,
    RunConfig, create_student, load_student, to_device, create_optimizer,
    set_max_steps_from_n_epochs, ema_update, create_tagged_metrics,
    get_n_tokens_per_sample,
    load_optimizer_checkpoint
)
from training.metrics import Aggregator
from training.dataloaders import InfDataLoader, ValidationDataLoader

@dataclass(kw_only=True)
class MyHyperparameters(Hyperparameters):
    temperature: float = 1.0  # Temperature for the distillation KL loss
    ema_alpha: float | None = None  # EMA decay, None means no EMA


def train_distil(
    rank,
    train_dataloader: InfDataLoader,
    val_dataloader: InfDataLoader | None,
    run_cfg: RunConfig,
    lp: MyHyperparameters,
) -> None:
    assert run_cfg.val_interval % run_cfg.log_interval == 0, "eval_interval must be divisible by log_interval"

    run = Run(run_cfg, lp)
    run.setup(rank)
    config = {
        "hyperparameters": lp.to_dict(),
        "train_data": train_dataloader.to_dict(),
        "val_data": val_dataloader.to_dict() if val_dataloader is not None else None,
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

    # If continuing training from an existing adapter, try to load optimizer state
    if run.run_cfg.model_cfg.student is not None:
        # Use the provided adapter checkpoint directory via model_id, as requested
        adapter_dir = Path(student_llm.adapter_ids[0])  # type: ignore[attr-defined]
        if run.run_cfg.optimizer_checkpointing:
            load_optimizer_checkpoint(run.global_rank, optimizer, adapter_dir)
    run.print_gpu_utilization()

    tokenizer = Tokenizer(str(base_llm.tokenizer_id))

    assert isinstance(run.run_cfg.compute_cfg.dp_size, int)
    len_trainset = train_dataloader.reset(run.dp_rank, run.run_cfg.compute_cfg.dp_size)
    iter_train_dataloader = iter(train_dataloader)

    lp.max_steps = max_steps = lp.max_steps or set_max_steps_from_n_epochs(
        len_trainset,
        micro_batch_size=train_dataloader.micro_batch_size,
        n_micro_batches_per_batch=train_dataloader.n_micro_batches_in_batch,
        n_devices=run_cfg.compute_cfg.dp_size,  # type: ignore
        n_epochs=lp.n_epochs,
        verbose=run.is_main_process,
    )

    # Validation data
    if val_dataloader is not None:
        val_dataloader.reset(run.dp_rank, run.run_cfg.compute_cfg.dp_size)

    adapter_name_to_export = "ema" if lp.ema_alpha is not None else "default"
    num_processed_samples = 0
    for step in range(max_steps):
        run.reset_step(step, max_steps)

        # Validation
        if (num_processed_samples % run_cfg.val_interval == 0) and  (val_dataloader is not None):
            if lp.ema_alpha is not None:
                student.set_adapter("ema")
                validate(
                    student, val_dataloader, rank, run,  # type: ignore
                    compute_ce=True, compute_reverse_kl=True,
                    compute_forward_kl=False,
                    name_prefix="val/ema")

            student.set_adapter("default")
            validate(
                student, val_dataloader, rank, run,  # type: ignore
                compute_ce=True, compute_reverse_kl=True,
                compute_forward_kl=False,
                name_prefix="val/default")

            if step > 0:
                run.export_adapters(
                    rank=run.global_rank, model=student, base_llm=base_llm,
                    hf_model_config=model_config, lora_config=peft_config,
                    device_mesh=run.device_mesh,
                    adapter_name_to_export=adapter_name_to_export)
                if run.run_cfg.optimizer_checkpointing:
                    run.save_optimizer_checkpoint(optimizer)

        run.reset_step_time()
        lr = lp.adjust_learning_rate(optimizer, step)
        run.add_to_metrics("train/lr", lr)

        student.train()
        # Distillation with reverse KL
        acc_distil = 0
        ema_acc_loss = 0
        aggregator = Aggregator(
            rank,
            dp_group=run.device_mesh.get_group(mesh_dim="dp"),
            dp_rank=run.dp_rank,
            tp_rank=run.tp_rank,
        )
        train_metrics = {}
        for _ in range(train_dataloader.n_micro_batches_in_batch):
            batch = next(iter_train_dataloader)  # type: ignore
            batch = to_device(batch, rank)
            run.print_gpu_utilization()
            T = lp.temperature
            # TODO: check shapes for different reductions
            loss_micro, _, _ = compute_kl(batch, student, temperature=T, reduction="sample")
            assert loss_micro is not None

            n_tokens_per_sample = get_n_tokens_per_sample(batch)

            name_prefix = "train/default"
            tagged_metrics = []

            tagged_metrics += create_tagged_metrics(
                batch["tags"], batch["drop"], loss_micro, n_tokens_per_sample, f"{name_prefix}/rkl")

            aggregator.add_batch(tagged_metrics)

            loss_micro = reduce_to_batch(loss_micro, batch['student_masks'])
            loss = loss_micro / train_dataloader.n_micro_batches_in_batch
            acc_distil += loss.item()
            loss.backward()

            # Calculate training loss for ema adapter
            if lp.ema_alpha is not None and run_cfg.compute_ema_train_loss:
                student.set_adapter("ema")
                with torch.no_grad():
                    loss_micro, _, _ = compute_kl(batch, student,
                        temperature=lp.temperature, reduction="batch")
                    assert loss_micro is not None
                    ema_loss = loss_micro / train_dataloader.n_micro_batches_in_batch
                    ema_acc_loss += ema_loss.item()
                student.set_adapter("default")

        num_processed_samples += train_dataloader.micro_batch_size * train_dataloader.n_micro_batches_in_batch * run_cfg.compute_cfg.dp_size  # type: ignore
        run.add_to_metrics("train/num_processed_samples", num_processed_samples)

        logit_metrics_total, logit_metrics_by_group = aggregator.get_average()
        train_metrics.update(logit_metrics_total)
        train_metrics.update(logit_metrics_by_group)
        run.add_dict_to_metrics(train_metrics)

        run.add_to_metrics("train/loss", acc_distil)
        if lp.ema_alpha is not None:
            run.add_to_metrics("train/loss/ema", ema_acc_loss)

        # Process other losses here
        if lp.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(student.parameters(), lp.max_grad_norm)

        optimizer.step()
        optimizer.zero_grad()

        # EMA update
        if lp.ema_alpha is not None:
            ema_update(student, alpha=lp.ema_alpha)

        # Release CUDA cache to reduce memory fragmentation
        torch.cuda.empty_cache()
        run.log_step()

        if run.is_checkpoint_step():
            run.export_adapters(
                rank=run.global_rank, model=student, base_llm=base_llm,
                hf_model_config=model_config, lora_config=peft_config,
                device_mesh=run.device_mesh,
                adapter_name_to_export=adapter_name_to_export)
            run.save_optimizer_checkpoint(optimizer)

    run.export_adapters(
        rank=run.global_rank, model=student, base_llm=base_llm,
        hf_model_config=model_config, lora_config=peft_config,
        device_mesh=run.device_mesh,
        adapter_name_to_export=adapter_name_to_export)
    if run.run_cfg.optimizer_checkpointing:
        run.save_optimizer_checkpoint(optimizer)

    run.print("Done")

