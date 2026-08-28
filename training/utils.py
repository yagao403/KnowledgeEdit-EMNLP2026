import os
import json
import pprint
import socket
import sys
import math
from dataclasses import dataclass, asdict
from functools import partial
import numpy as np
from pathlib import Path
import random
import re
import time
from typing import Literal, Protocol, Callable
import shutil
from collections import Counter

import torch
import torch.distributed as dist
from torch.utils.data import Sampler
from torch.utils.data.dataloader import DataLoader
from torch.distributed.device_mesh import init_device_mesh, DeviceMesh
from transformers import AutoModelForCausalLM
from peft import get_peft_model

import wandb

from core import DATA_PATH
from core.llm import LLM
from core.tokenizer import Tokenizer
from core.putils import print_gpu_utilization, print_gpu_utilization_cuda
from core.utils import DualOutput
from training.st_dataset import STDataset, ConcatSTDataset, STStepMessages
from training.losses import compute_kl, compute_ce_loss, get_n_tokens_per_sample
from training.metrics import Aggregator, TaggedMetric


from peft import LoraConfig, PeftConfig, PeftModel, PeftMixedModel
from transformers import AutoConfig
import tensor_parallelism.core_model as tpm
from tensor_parallelism.utils import (
    str_to_dtype,
    get_nested_attribute,
    copy_directory_contents,
    move_adapters_to_cpu_for_FSDP_TP,
    move_adapters_to_cpu_for_FSDP2,
)

def clean_xml_content(filename):
    with open(filename, 'r', encoding='utf-8', errors='ignore') as file:
        content = file.read()

    # XML 1.0 allows: \t, \n, \r, and space through \uD7FF, \uE000 through \uFFFD, excluding the surrogate block \uD800 through \uDFFF
    # Characters not allowed in XML 1.0 are \u0000 through \u0008, \u000B, \u000C, \u000E through \u001F, and others above \uFFFD
    # We remove any characters not allowed in XML 1.0
    cleaned_content = re.sub(r"[\u0000-\u0008\u000B\u000C\u000E-\u001F\uD800-\uDFFF\uFFFE\uFFFF]", "", content)

    # Save the cleaned content back to a temporary file or overwrite the original
    temp_filename = filename + '.cleaned'
    with open(temp_filename, 'w', encoding='utf-8') as file:
        file.write(cleaned_content)

    return temp_filename

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using CUDA
    np.random.seed(seed)
    random.seed(seed)


def print_memory_usage(description):
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    print(f"{description}:")
    print(f"Allocated memory: {allocated / 1024 ** 3:.2f} GB")
    print(f"Reserved memory: {reserved / 1024 ** 3:.2f} GB")


def handle_oom(
    batch,
    strict_memory: bool=True,
    teacher_key='teacher_seqs'
):
    if strict_memory:
        raise
    print("WARNING - Out Of Memory Error. Skipping batch due to memory constraints")
    print("Batch dimensions:")
    print("Teacher inputs", batch[teacher_key][..., :-1].shape)
    print("Student inputs", batch['student_seqs'][..., :-1].shape)
    print_memory_usage("Current memory status")
    del batch
    torch.cuda.empty_cache()
    print_memory_usage("Memory status after clearing memory")
    loss = torch.zeros((1))
    return loss, loss

# All LoRA sub-module tags we want to wipe out
LORA_TAGS = (
    "lora_A", "lora_B",
    "lora_embedding_A", "lora_embedding_B",
)

def lora_to_base_name(param_name: str) -> str:
    """
    Given something like
      'model.layers.0.self_attn.q_proj.lora_A.default.weight'
    return
      'model.layers.0.self_attn.q_proj.weight'
    """
    parts = param_name.split(".")
    for i, p in enumerate(parts):
        if p in LORA_TAGS:
            # keep everything *before* the tag and the final 'weight' / 'bias'
            return ".".join(parts[:i] + parts[-1:])
    raise ValueError(f"{param_name} is not a LoRA parameter")


@dataclass
class _GroupedDataMixin:
    """
    Mixin providing shared functionality for data configs that carry
    `group_xml_paths` and `micro_batch_size`.

    Child classes can override `_extra_keys_to_pop_on_flatten` to remove
    additional keys when the data has a single group and is flattened.
    """
    group_xml_paths: dict[str, list[Path]]  # Dictionary group_names -> list of xml filepaths
    micro_batch_size: int

    # --- Overridables -----------------------------------------------------
    def _extra_keys_to_pop_on_flatten(self) -> list[str]:
        return []

    def to_dict(self):
        d = self.__dict__.copy()
        d["group_xml_paths"] = {
            k: group_files(get_relative_path(v))
            for k, v in self.group_xml_paths.items()
        }
        if len(self.group_xml_paths) == 1:
            d["xml_paths"] = list(d["group_xml_paths"].values())[0]
            d.pop("group_xml_paths")
            # Allow subclasses to drop keys that don't make sense after flattening
            for k in self._extra_keys_to_pop_on_flatten():
                d.pop(k, None)
        return d

    def check_files_exist(self, description: str = ""):
        if description:
            print(f"Checking files for {description} ... ")
        for group_name, xml_paths in self.group_xml_paths.items():
            print(f"  Group '{group_name}': {len(xml_paths)} files", end=" ... ")
            assert len(xml_paths) > 0, f"No XML files provided for {group_name} group"
            for xml_path in xml_paths:
                assert xml_path.exists(), f"Training XML file does not exist: {xml_path}"
            print("ok")

    @property
    def xml_paths(self):
        return sum(self.group_xml_paths.values(), [])

    @property
    def tags_per_xml(self) -> list[set[str]] | None:
        # The group name is used as the tag for all files in that group
        if len(self.group_xml_paths) == 1:
            # If there's only one group, we do not use tags by default
            # If add_filename_tags is True in get_val_data_loader, filename tags will be added
            return None
        tags_per_xml = sum(
            (
                [{group_name}]*len(paths)
                for group_name, paths in self.group_xml_paths.items()
            ), []
        )
        return tags_per_xml

@dataclass
class TrainingData(_GroupedDataMixin):
    n_micro_batches_in_batch: int
    max_teacher_seq_len: int
    student_dropout_rate: float = 0.9  # makes sense only for logit_loss
    group_mix_props: dict[str, float] | None = None  # Mixing proportions, e.g., {"class_1": 0.9, "class_2": 0.1}
    group_loss_weights: dict[str, float] | None = None  # e.g., {"class_1": 1, "class_2": 10}
    add_filename_tags: bool = True  # Whether to add filename-based tags to the samples
    tags_assigner: Callable[[STStepMessages], set[str]] | None = None  # Function to assign tags based on file path

    @classmethod
    def with_single_group(
        cls,
        xml_paths: list[Path],
        micro_batch_size: int,
        n_micro_batches_in_batch: int,
        max_teacher_seq_len: int,
        student_dropout_rate: float,
        add_filename_tags: bool = True,
        tags_assigner: Callable[[STStepMessages], set[str]] | None = None,
    ):
        return cls(
            group_xml_paths={"all": xml_paths},
            micro_batch_size=micro_batch_size,
            n_micro_batches_in_batch=n_micro_batches_in_batch,
            max_teacher_seq_len=max_teacher_seq_len,
            student_dropout_rate=student_dropout_rate,
            group_mix_props={"all": 1.0},
            group_loss_weights=None,
            add_filename_tags=add_filename_tags,
            tags_assigner=tags_assigner,
        )

    def __post_init__(self):
        if self.group_mix_props is None:
            if len(self.group_xml_paths) > 1:
                raise ValueError("group_mix_props must be provided when data consists of multiple groups")
            self.group_mix_props = {group: 1.0 for group in self.group_xml_paths.keys()}

    # Ensure TrainingData drops `group_mix_props` when flattening to a single group
    def _extra_keys_to_pop_on_flatten(self) -> list[str]:
        return ["group_mix_props"]

@dataclass
class ValidationData(_GroupedDataMixin):
    add_filename_tags: bool = True  # Whether to add filename-based tags to the samples
    tags_assigner: Callable[[STStepMessages], set[str]] | None = None

    @classmethod
    def with_single_group(
        cls,
        xml_paths: list[Path],
        micro_batch_size: int,
        add_filename_tags: bool = True,
        group_name: str = "all",
        tags_assigner: Callable[[STStepMessages], set[str]] | None = None,
    ):
        return cls(
            group_xml_paths={group_name: xml_paths},
            micro_batch_size=micro_batch_size,
            add_filename_tags=add_filename_tags,
            tags_assigner=tags_assigner,
        )

@dataclass
class ModelConfig:
    base_model: LLM
    student: LLM | None = None  # Specify an existing adapter's name to continue training
    teacher_type: Literal["student", "student_base", "model"] | None = "student_base"

    lora_target_modules: Literal["full", "qk", "qv", "full-no-lm-head"] = "full"
    lora_r: int = 512

    def to_dict(self):
        d = {
            "base_model": self.base_model.to_dict(),
            "student": self.student.to_dict() if self.student is not None else None,
            "lora_r": self.lora_r,
            "lora_target_modules": self.lora_target_modules,
        }
        return d

@dataclass
class ComputeConfig:
    tp_size: int = 1  # Tensor parallelism size
    dp_size: int | Literal["auto"] = "auto"  # Data parallelism size. If None, it will be computed in post_init
    world_size: int = 1  # Total number of GPUs used for training
    node_id: int = 0  # Index of the current node (0-indexed)
    host_node: str | None = None  # Name of the host node (i.e. head node). This must be specified in multi-node training is used
    ip: str = "localhost"  # IP address of the host node. This must be specified in multi-node training is used
    multi_node: bool = False  # Whether doing multi-node training
    gradient_checkpointing: bool = True  # Enable gradient checkpointing
    use_fsdp_only: bool = True  # This is used to force FSDP-only training. Set to True use FSDP2 only
    torch_dtype: str = 'bf16'  # Model dtype (i.e. bf16 or f32)
    transformer_layer_cls: type[torch.nn.Module] | tuple[type[torch.nn.Module]] | None = None  # The transformer layer class (e.g., LlamaDecoderLayer, Qwen2DecoderLayer).
                                                   # This is will be used to manually apply gradient checkpointing, or any operations
                                                   # that need this class name.
    def __post_init__(self):
        # Setup DP size
        if not (isinstance(self.dp_size, int) or self.dp_size == "auto"):
            raise TypeError("dp_size must be an int or 'auto'")
        if isinstance(self.dp_size, int):
            assert self.tp_size * self.dp_size == self.world_size, "world_size must be equal to tp_size * dp_size"
        elif self.dp_size == "auto":
            if self.use_fsdp_only:
                assert self.tp_size == 1, "tp_size > 1 detected when forcing FSDP-only. It should be set to 1."
                self.dp_size = self.world_size
            else:
                assert self.world_size % self.tp_size == 0, f"DP size must be divisible by TP size, current DP size: {self.world_size}, TP size: {self.tp_size}"
                self.dp_size = self.world_size // self.tp_size

        # Setup multiple node training
        self.ip = "localhost"  # TODO: make it default
        if self.multi_node:
            if self.node_id != 0:
                assert self.host_node is not None, "Host node must be provided for non-zero node_id"
                self.ip = get_ip_address(self.host_node)
                print(f"Host node: {self.host_node} - IP address: {self.ip}")

@dataclass
class Hyperparameters:
    max_lr: float
    weight_decay: float
    max_steps: int | None = None  # If provided, n_epochs is ignored
    max_grad_norm: float = 1.0  # This seems to produce better results
    lr_decay: bool = False
    warmup_steps: int = 0  # None means using warmup_ratio
    warmup_ratio: float | None = None
    logit_loss_threshold: float | None = None
    n_epochs: int = 0  # 0 means using max_steps
    lora_a_weight_decay_only: bool = False  # Whether to apply weight decay only to the adapter A of LoRA
    lora_b_weight_decay_only: bool = False  # Whether to apply weight decay only to the adapter B of LoRA

    def adjust_learning_rate(self, optimizer, step: int):
        assert self.max_steps is not None
        max_steps = self.max_steps
        warmup_steps = self.warmup_steps

        lr = self.max_lr
        if step <= warmup_steps and warmup_steps:
            # linear warmup
            lr = self.max_lr * step / warmup_steps
        else:
            if self.lr_decay:
                multiplier = max(0.0, float(max_steps - step) / float(max(1, max_steps - warmup_steps)))
                lr = self.max_lr * multiplier

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        return lr

    def to_dict(self) -> dict:
        return asdict(self)

def set_max_steps_from_n_epochs(
    len_dataset: int,
    micro_batch_size: int,
    n_micro_batches_per_batch: int,
    n_devices: int,
    n_epochs: int,
    verbose: bool = False,
) -> int:
    assert n_epochs > 0, "n_epochs must be > 0"
    n_batches = math.ceil(
        len_dataset / n_micro_batches_per_batch
        / micro_batch_size / n_devices
    )
    max_steps = n_epochs * n_batches
    if verbose:
        print(f"Training for {n_epochs} epochs, {max_steps} steps", flush=True)

    return max_steps

@dataclass(kw_only=True)
class RunConfig:
    project_name: str
    run_name: str
    project_path: Path
    wnb_project: str
    group_name: str

    model_cfg: ModelConfig
    compute_cfg: ComputeConfig

    adapter_name: str = 'default'  # Name of the adapter to use. This is used to load the adapter from the adapter_dir.
    adapter_dir: str = 'adapters'  # Directory to save/load adapters. This is set in Run.setup()

    seed: int = 42  # Random seed to initialize the samplers for Training and Validation Data

    val_interval: int = 0  # zero means no validation
    log_interval: int = 1
    checkpoint_interval: int = 0  # zero means no checkpointing
    latest_checkpoint_interval: int = 0  # zero means no checkpointing
    use_wandb: bool = False
    optimizer_checkpointing: bool = True

    # vLLM eval
    eval_interval: int = 0  # zero means no evaluation
    evaluation_job: str | None = None # Path to the evaluation job python script

    compute_ema_train_loss: bool = False # Whether to compute the training loss for the EMA adapter

    # Batch sampling mode
    sampling_mode: Literal["standard", "batch_optimized", "episodic"] = "standard"

    notes: str | None = None  # Notes to be added on wandb

    @property
    def run_path(self):
        return self.project_path / self.run_name

    def to_dict(self):
        d = self.__dict__.copy()
        d["model_cfg"] = self.model_cfg.to_dict()
        d["compute_cfg"] = asdict(self.compute_cfg)
        d["run_path"] = str(self.run_path)

        # Convert any Path objects to strings
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
        return d

class Run:
    def __init__(
        self,
        run_cfg: RunConfig,
        hp: Hyperparameters,
    ):
        self.run_cfg = run_cfg
        self.hp = hp

        # These are set in setup()
        self.is_main_process = False
        self.is_rocm = "rocm" in torch.__version__
        self.devices = 1

        self.metrics = {}
        self.step = 0
        self.max_steps = None
        self.step_t0 = 0
        self.last_eval_step = None
        self.num_processed_samples = 0

        self.wandb_run = None

    @property
    def run_path(self):
        return self.run_cfg.project_path / self.run_cfg.run_name

    def setup(self, rank: int):
        run_cfg = self.run_cfg
        self.run_cfg.adapter_dir = str(self.run_path)

        self.rank = rank
        self.tp_size = 1

        if self.run_cfg.compute_cfg.multi_node:
            # We assume 8 GPUs per node
            self.global_rank = self.run_cfg.compute_cfg.node_id * 8 + self.rank
        else:
            self.global_rank = rank

        print(f"global_rank: {self.global_rank} {self.rank=} {self.run_cfg.compute_cfg.ip=}")
        self.is_main_process = self.global_rank==0

        # Setup environment for distributed training
        os.environ['MASTER_ADDR'] = self.run_cfg.compute_cfg.ip
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
        if self.run_cfg.compute_cfg.multi_node:
            dist.init_process_group("nccl", rank=self.global_rank, world_size=self.run_cfg.compute_cfg.world_size)
        else:
            dist.init_process_group("nccl", rank=rank, world_size=self.run_cfg.compute_cfg.world_size)
        torch.cuda.set_device(rank)

        assert isinstance(self.run_cfg.compute_cfg.dp_size, int)
        if self.run_cfg.compute_cfg.use_fsdp_only:
            self.device_mesh = init_device_mesh("cuda", (self.run_cfg.compute_cfg.dp_size,), mesh_dim_names=("dp",))
            self.dp_rank = self.device_mesh["dp"].get_local_rank()
            self.tp_rank = 0 # This is a hack to ensure backward compatibility
        else:
            self.device_mesh = init_device_mesh("cuda", (self.run_cfg.compute_cfg.dp_size, self.run_cfg.compute_cfg.tp_size), mesh_dim_names=("dp", "tp"))
            self.tp_rank = self.device_mesh["tp"].get_local_rank()
            self.dp_rank = self.device_mesh["dp"].get_local_rank()

        if self.is_main_process:
            self.make_run_dir()
        self.log_to_wandb = self.is_main_process and self.run_cfg.use_wandb

        self.devices = self.run_cfg.compute_cfg.dp_size
        # self.setup_wandb()
        self.print("run_name:", run_cfg.run_name)
        self.print("Hyperparameters:")
        self.pprint(self.hp)

        return None

    def print(self, *args, **kwargs):
        kwargs["flush"] = True
        if self.is_main_process:
            print(*args, **kwargs)

    def pprint(self, *args, **kwargs):
        if self.is_main_process:
            pprint.pprint(*args, **kwargs)

    def print_gpu_utilization(self):
        if self.is_main_process:
            print_gpu_utilization_cuda() if not self.is_rocm else print_gpu_utilization()

    def make_run_dir(self):
        os.makedirs(self.run_path, exist_ok=True)

    def set_main_process(self, is_main_process):
        self.is_main_process = is_main_process
        self.log_to_wandb = is_main_process and self.run_cfg.use_wandb

    def setup_print_output(self, process_index):
        if self.is_main_process:
            output_log = self.run_path / f"output_{process_index}.log"
            # Set the output to the console and to a file
            sys.stdout = DualOutput(output_log)

    def setup_wandb(self):
        assert isinstance(self.hp, Hyperparameters)
        run_cfg = self.run_cfg
        if self.is_main_process and run_cfg.use_wandb:
            training_data_paths = sum(self.hp.training_data.values(), [])
            training_data = get_relative_path(training_data_paths)

            validation_data = get_relative_path(self.hp.validation_data)

            wandb_config = {
                "hyperparameters": asdict(self.hp),
                "run_cfg": self.run_cfg.to_dict(),
                "model": self.run_cfg.model_cfg.to_dict(),
                "training_data": group_files(training_data),
                "validation_data": group_files(validation_data),
            }

            wandb.init(
                project=run_cfg.project_path.name,
                name=run_cfg.run_name,
                group=run_cfg.group_name,
                allow_val_change=True,
                config=wandb_config,
                notes=run_cfg.notes,
            )
            # wandb.define_metric("num_processed_samples")

    def setup_wandb_new(
        self,
        config: dict,
    ):
        config = config | {
            "run_cfg": self.run_cfg.to_dict(),
            "model": self.run_cfg.model_cfg.to_dict(),
        }
        if self.is_main_process and self.run_cfg.use_wandb:
            wandb.init(
                project=self.run_cfg.wnb_project or self.run_cfg.project_path.name,
                name=self.run_cfg.run_name,
                group=self.run_cfg.group_name,
                allow_val_change=True,
                config=config,
                notes=self.run_cfg.notes,
            )
            print("Wandb config:", config)


    def is_progress_checkpoint_step(self):
        return self.run_cfg.checkpoint_interval and ((self.step) % self.run_cfg.checkpoint_interval == (self.run_cfg.checkpoint_interval - 1) or self.step == 0)

    def is_latest_checkpoint_step(self):
        return self.run_cfg.latest_checkpoint_interval and (self.step) % self.run_cfg.latest_checkpoint_interval == (self.run_cfg.latest_checkpoint_interval - 1)

    def is_checkpoint_step(self):
        return self.is_progress_checkpoint_step() or self.is_latest_checkpoint_step()

    def reset_step(self, step: int, max_steps: int):
        self.step = step
        self.max_steps = max_steps
        self.metrics = {}

    def reset_step_time(self):
        self.step_t0 = time.perf_counter()

    @property
    def is_logging_step(self):
        return self.is_main_process and (self.step % self.run_cfg.log_interval == 0)

    def add_to_metrics(self, key: str, value):
        # Note: If value is a pytorch tensor, pass it instead of value.item()

        # We log only if we are in the logging step
        if not self.is_logging_step:
            return

        if isinstance(value, torch.Tensor):
            value = value.item()  # expensive device-to-host synchronization

        self.metrics[key] = value

    def add_dict_to_metrics(self, d: dict):
        # Note: If value is a pytorch tensor, pass it instead of value.item()
        # We log only if we are in the logging step
        if not self.is_logging_step:
            return

        d = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in d.items()}
        self.metrics.update(d)

    def log_step(self):
        if not self.is_logging_step:
            return

        # Note: we log the one-batch loss for the main process
        # To gather losses across devices, we should do something like
        # losses = accelerator.gather(loss)
        # loss_item = losses.mean().item()
        t1 = time.perf_counter()
        s_metrics = [
            f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}"
            for k, v in self.metrics.items()
        ]
        print(
            f"[{self.step}]: " +
            (" ".join(s_metrics)) +
            f"[{(t1 - self.step_t0) * 1000:.2f}ms]",
            flush=True
        )
        if self.log_to_wandb:
            self.metrics["train/step_time"] = (t1 - self.step_t0) * 1000
            wandb.log(self.metrics, self.step, commit=True)

    def log_evaluation(self, metrics):
        if self.log_to_wandb:
            wandb.log(self.metrics, self.last_eval_step)

    def save_base_model_config(self, base_llm, run_path: Path | None = None):
        if self.is_main_process:
            run_path = self.run_path if run_path is None else run_path
            with open(run_path / "base_model_config.json", 'w', encoding='utf-8') as f:
                json.dump(base_llm.to_dict(), f, ensure_ascii=False, indent=4)
            self.print(f"Saved to {run_path}")

    def export_adapters(
        self,
        rank,
        model,
        base_llm: LLM,
        hf_model_config,
        lora_config,
        device_mesh,
        adapter_name_to_export: str = 'default',
    ):
        '''
        The lora_config in this function is indeed LoraConfig object instead of the dataclass lora_config
        '''
        if not (self.is_progress_checkpoint_step() or self.is_latest_checkpoint_step()):
            return
        print('Start exporting adapters ...')
        global_rank = rank
        # Load model into cpu and extract state_dict and buffers
        torch_dtype = str_to_dtype(self.run_cfg.compute_cfg.torch_dtype)

        cpu_model = AutoModelForCausalLM.from_pretrained(
            base_llm.model_id,
            config=hf_model_config,
            # device_map='cpu' if global_rank == 0 else 'meta',
            device_map='meta',
            load_in_8bit=False,
            torch_dtype=torch_dtype,
            #attn_implementation="flash_attention_2"
            trust_remote_code=True,
            # cache_dir=os.environ['HF_HOME'],
        )
        cpu_model = get_peft_model(model=cpu_model,
                                peft_config=lora_config,
                                adapter_name='default',
                                autocast_adapter_dtype=False)

        # If global rank is 0, materialize the peft weights to prepare for
        # copying trained peft weights from distributed model into cpu model for saving
        if global_rank == 0:
            peft_modules = ['.'.join(weight_name.split('.')[:-1]) for weight_name in cpu_model.state_dict() if 'lora' in weight_name]
            # Materialize only peft weights
            for weight_name in peft_modules:
                peft_module = get_nested_attribute(cpu_model, weight_name)
                # make sure they also have torch_dtype type
                _to_empty = getattr(peft_module, "to_empty", None)
                if callable(_to_empty):
                    _to_empty(device=torch.device("cpu"), recurse=False)
                peft_module.to(dtype=torch_dtype)

        # Move adapters from the sharded models to the cpu model
        if self.run_cfg.compute_cfg.use_fsdp_only:
            move_adapters_to_cpu_for_FSDP2(
                rank=rank,
                model=model,
                cpu_state_dict=cpu_model.state_dict(),
                adapter_name_to_export=adapter_name_to_export,
            )
        else:
            move_adapters_to_cpu_for_FSDP_TP(rank=rank,
                                            model=model,
                                            cpu_state_dict=cpu_model.state_dict(),
                                            device_mesh=device_mesh,
                                            )

        # Save cpu adapters
        if global_rank == 0:
            peft_state_dict = {weight_name: weight for weight_name, weight in cpu_model.state_dict().items() if 'lora' in weight_name}
            temp_dir = self.run_path.parent / (self.run_path.name + '_temp')
            progress_ckpt_folder = self.run_path / f'step_{self.step}'
            latest_ckpt_folder = self.run_path / 'latest'

            try:
                # First save to temporary directory
                cpu_model.save_pretrained(
                    str(temp_dir),
                    state_dict=peft_state_dict,
                )  # at this point if timeout, the old adapters are still in the adapter_dir

                # At this point if timeout, the new adapters are already in the temp_dir
                if self.is_progress_checkpoint_step():
                    copy_directory_contents(temp_dir, progress_ckpt_folder)
                if self.is_latest_checkpoint_step():
                    copy_directory_contents(temp_dir, latest_ckpt_folder)

                # Clean up temporary directory
                shutil.rmtree(temp_dir)  # at this point if timeout, the adapters are already in the self.run_path
                print(f'Adapters saved to {latest_ckpt_folder or progress_ckpt_folder}')

            except Exception as e:
                print(f"Error saving adapters: {e}")
                # Clean up temp dir in case of error
                shutil.rmtree(temp_dir, ignore_errors=True)
                raise

            # Store the base model config in the step directory as well if it's also a progress checkpointing step
            if self.is_progress_checkpoint_step():
                self.save_base_model_config(base_llm, progress_ckpt_folder)
                print(f'Adapters saved to {progress_ckpt_folder}')

            # copy the base model config to the adapter dir (i.e. run_path)
            if self.is_latest_checkpoint_step():
                self.save_base_model_config(base_llm, latest_ckpt_folder)
                print(f'Latest adapters saved to {latest_ckpt_folder}')

    def save_optimizer_checkpoint(
        self,
        optimizer: torch.optim.Optimizer,
    ) -> Path | None:
        """
        Save optimizer state alongside adapters.

        Strategy (simple and robust for FSDP2):
        - Save a per-rank optimizer state_dict into the same directory where adapters are exported:
        run_path/step_{step}/optimizer.rank{global_rank}.pt and/or run_path/latest/optimizer.rank{global_rank}.pt
        - Assumes continuation uses the same parallelism layout so each rank reloads its own shard.
        """
        latest_ckpt_path = None
        progress_ckpt_path = None

        # Progress checkpoint directory
        if self.is_progress_checkpoint_step():
            step_dir = self.run_path / f"step_{self.step}"
            step_dir.mkdir(parents=True, exist_ok=True)
            progress_ckpt_path = step_dir / f"optimizer.rank{self.global_rank}.pt"
            torch.save(optimizer.state_dict(), progress_ckpt_path)
            print(f"Optimizer checkpoint at step {self.step} is saved to {progress_ckpt_path}", flush=True)

        # Latest checkpoint directory
        if self.is_latest_checkpoint_step():
            latest_dir = self.run_path / "latest"
            latest_dir.mkdir(parents=True, exist_ok=True)
            latest_ckpt_path = latest_dir / f"optimizer.rank{self.global_rank}.pt"
            torch.save(optimizer.state_dict(), latest_ckpt_path)
            print(f"Latest optimizer checkpoint at step {self.step} is saved to {latest_ckpt_path}", flush=True)

        return progress_ckpt_path or latest_ckpt_path

def load_optimizer_checkpoint(
    global_rank: int,
    optimizer: torch.optim.Optimizer,
    adapter_ckpt_dir: Path,
):
    """
    Load the optimizer state for the current rank from the provided adapter
    checkpoint directory. This function assumes the optimizer state was saved
    per-rank with the name pattern: optimizer.rank{global_rank}.pt

    Returns True if successfully loaded, otherwise False.
    """
    ckpt_path = adapter_ckpt_dir / f"optimizer.rank{global_rank}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Optimizer checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location="cpu")
    optimizer.load_state_dict(state)
    print(f"Loaded optimizer state from {ckpt_path}")

class SupportsToDict(Protocol):
    def to_dict(self) -> dict: ...

def setup_wandb(
    train_ce_data: TrainingData,
    train_kl_data: TrainingData,
    val_ce_data: ValidationData,
    val_kl_data: ValidationData,
    model_config: ModelConfig,
    run_cfg: RunConfig,
    lp: SupportsToDict,
):
    if run_cfg.use_wandb:
        config = {
            "learning": lp.to_dict(),
            "model": model_config.to_dict(),
            "train_ce_data": train_ce_data.to_dict(),
            "train_kl_data": train_kl_data.to_dict(),
            "val_ce_data": val_ce_data.to_dict(),
            "val_kl_data": val_kl_data.to_dict(),
            "run_cfg": run_cfg.to_dict(),
        }
        wandb.init(
            project=run_cfg.project_path.name,
            name=run_cfg.run_name,
            group=run_cfg.group_name,
            allow_val_change=True,
            config=config,
            notes=run_cfg.notes,
        )
        print("Wandb config:", config)


def create_student(
    rank: int,
    device_mesh: DeviceMesh,
    use_ema: bool,
    model_cfg: ModelConfig,
    compute_cfg: ComputeConfig
):
    base_llm: LLM = model_cfg.base_model

    if model_cfg.lora_target_modules == "full":
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ]
    elif model_cfg.lora_target_modules == "qk":
        target_modules=["q_proj", "k_proj"]
    elif model_cfg.lora_target_modules == "qv":
        target_modules = ["q_proj", "v_proj"]
    elif model_cfg.lora_target_modules == "full-no-lm-head":
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ]
    else:
        raise ValueError(f"Invalid lora_target_modules: {model_cfg.lora_target_modules}")
    peft_config = LoraConfig(
        r=model_cfg.lora_r,
        lora_alpha=model_cfg.lora_r*2,
        target_modules=target_modules,
        lora_dropout=0.05,
    )

    # Setup model configuration
    model_config = AutoConfig.from_pretrained(base_llm.model_id)
    model_config.sharded_ratio = 1/compute_cfg.tp_size if compute_cfg.tp_size is not None else 1
    model_config.gradient_checkpointing = compute_cfg.gradient_checkpointing
    model_config.use_cache = False
    model_config._attn_implementation = "flash_attention_2"
    if 'text_config' in model_config:
        model_config.text_config.use_cache = False

    if compute_cfg.use_fsdp_only:
        print("Use FSDP2", flush=True)

        # Important: forcibly disable gradient checkpointing for FSDP2 to prevent the model to handle it by itself,
        # we will apply our own gradient checkpointing policy later using `manually_apply_gradient_checkpointing` flag in `load_base_model_without_adapters`
        model_config.gradient_checkpointing = False
        if 'text_config' in model_config:
            model_config.text_config.gradient_checkpointing = False
        if 'vision_config' in model_config:
            model_config.vision_config.gradient_checkpointing = False

        assert len(base_llm.adapter_ids) <= 1, "Only one adapter is supported at the moment"

        assert device_mesh is not None
        lora_model = tpm.create_FSDP2_parallelized_model(
            rank=rank,
            device_mesh=device_mesh,
            model_id=base_llm.model_id,
            model_config=model_config,
            adapter_id=base_llm.adapter_ids[0] if base_llm.adapter_ids else None,
            adapter_name='default',
            torch_dtype=str_to_dtype(compute_cfg.torch_dtype),
            peft_config=peft_config,
            gradient_checkpointing=compute_cfg.gradient_checkpointing,
            transformer_layer_cls=compute_cfg.transformer_layer_cls if compute_cfg.gradient_checkpointing else None,
            device_map='meta',
            training=True,
            efficient_loading=True,
            merge=True,
            ema=use_ema,
        )
        assert isinstance(lora_model, (PeftModel, PeftMixedModel))
    else:
        raise ValueError(
            "Only FSDP2 is supported for training with PyTorch. "
            "Please set `use_fsdp_only=True` in the RunConfig."
        )

    lora_model.train()

    # Release cuda cache to reduce memory fragmentation
    torch.cuda.empty_cache()
    return lora_model, model_config, peft_config

def load_student(
    rank: int,
    device_mesh: DeviceMesh,
    use_ema: bool,
    model_cfg: ModelConfig,
    compute_cfg: ComputeConfig,
):
    assert model_cfg.student is not None
    student_llm: LLM = model_cfg.student
    adapter_id = student_llm.adapter_ids[0]
    print(f"Continue training adapter: {adapter_id} on base model: {student_llm.model_id}", flush=True)

    model_id = student_llm.model_id

    # Setup model configuration
    model_config = AutoConfig.from_pretrained(model_id)
    model_config.sharded_ratio = 1/compute_cfg.tp_size
    model_config.gradient_checkpointing = compute_cfg.gradient_checkpointing
    model_config.use_cache = False
    model_config._attn_implementation = "flash_attention_2"
    if 'text_config' in model_config:
        model_config.text_config.use_cache = False

    if compute_cfg.use_fsdp_only:
        print("Use FSDP2", flush=True)

        # Important: forcibly disable gradient checkpointing for FSDP2 to prevent the model to handle it by itself,
        # we will apply our own gradient checkpointing policy later using `manually_apply_gradient_checkpointing` flag in `load_base_model_without_adapters`
        model_config.gradient_checkpointing = False
        if 'text_config' in model_config:
            model_config.text_config.gradient_checkpointing = False
        if 'vision_config' in model_config:
            model_config.vision_config.gradient_checkpointing = False

        peft_config = PeftConfig.from_pretrained(adapter_id)
        peft_config.inference_mode = False  # IMPORTANT: Set inference_mode to False to enable trainable parameters
        # print(f'{peft_config=} {type(peft_config)=}')
        lora_model = tpm.create_FSDP2_parallelized_model(
            rank=rank,
            device_mesh=device_mesh,
            model_id=model_id,
            model_config=model_config,
            adapter_id=adapter_id,
            adapter_name='default',
            torch_dtype=str_to_dtype(compute_cfg.torch_dtype),
            peft_config=peft_config,
            gradient_checkpointing=compute_cfg.gradient_checkpointing,
            transformer_layer_cls=compute_cfg.transformer_layer_cls if compute_cfg.gradient_checkpointing else None,
            device_map='meta',
            training=True,
            efficient_loading=True,
            merge=False,
            ema=use_ema,
        )
        assert isinstance(lora_model, (PeftModel, PeftMixedModel))
    else:
        raise ValueError(
            "Only FSDP2 is supported for training with PyTorch. "
            "Please set `use_fsdp_only=True` in the RunConfig."
        )

    lora_model.train()

    # run.print(model)
    print(f"Number of trainable parameters: {num_parameters(lora_model, requires_grad=True):,}", flush=True)
    print(f"Number of fixed parameters: {num_parameters(lora_model, requires_grad=False):,}", flush=True)
    print_gpu_utilization()

    # Release cuda cache to reduce memory fragmentation
    torch.cuda.empty_cache()
    return lora_model, model_config, peft_config

def to_device(batch, device):
    for key in batch:
        if key not in ("tags", "drop"):
            batch[key] = batch[key].to(device)
    return batch

def slice_batch(batch, start: int, end: int):
    sliced_batch = {}
    for key in batch:
        sliced_batch[key] = batch[key][start:end]
    return sliced_batch

def create_tagged_metrics(
    list_of_tags: list[set[str]],
    drop: list[bool],
    metric: torch.Tensor | None,  # (batch_size,)
    n_tokens_per_sample: torch.Tensor,  # (batch_size,)
    name: str,
) -> list[TaggedMetric]:
    if metric is None:
        return []
    return [
        TaggedMetric(tags, name, m.item(), drop, int(n_tokens.item()))
        for tags, drop, m, n_tokens in zip(
            list_of_tags, drop, metric, n_tokens_per_sample, strict=True)
    ]

def validate(
    student,
    dataloader,
    rank: int,
    run: Run,
    compute_reverse_kl: bool,
    compute_forward_kl: bool,
    compute_ce: bool,
    name_prefix: str = "val",
):
    if run.is_main_process:
        assert run.is_logging_step, "Validation should only be done at logging steps"

    student.eval()
    metrics = {}
    with torch.no_grad():
        t0 = time.perf_counter()
        if dataloader is not None:
            aggregator = Aggregator(
                rank,
                dp_group=run.device_mesh.get_group(mesh_dim="dp"),
                dp_rank=run.dp_rank,
                tp_rank=run.tp_rank,
            )

            for batch in dataloader:
                batch = to_device(batch, rank)

                n_tokens_per_sample = get_n_tokens_per_sample(batch)
                rkl, fkl, ce = None, None, None
                if compute_reverse_kl or compute_forward_kl:
                    rkl, fkl, ce = compute_kl(
                        batch, student, temperature=1, reduction="sample",
                        compute_reverse_kl=compute_reverse_kl,
                        compute_forward_kl=compute_forward_kl,
                        compute_ce=compute_ce,
                    )
                else:
                    ce = compute_ce_loss(batch, student, reduction="sample")

                tagged_metrics = []
                tagged_metrics += create_tagged_metrics(
                    batch["tags"], batch["drop"], rkl, n_tokens_per_sample, f"{name_prefix}/rkl")
                tagged_metrics += create_tagged_metrics(
                    batch["tags"], batch["drop"], fkl, n_tokens_per_sample, f"{name_prefix}/fkl")
                tagged_metrics += create_tagged_metrics(
                    batch["tags"], batch["drop"], ce, n_tokens_per_sample, f"{name_prefix}/ce")
                aggregator.add_batch(tagged_metrics)

            logit_metrics_total, logit_metrics_by_group = aggregator.get_average()
            metrics.update(logit_metrics_total)
            metrics.update(logit_metrics_by_group)

    metrics[f"{name_prefix}/time"] = time.perf_counter() - t0
    run.add_dict_to_metrics(metrics)
    run.print(f"Validation results ({name_prefix}):", metrics)

def create_optimizer(
    model,
    lora_a_weight_decay_only: bool = False,
    lora_b_weight_decay_only: bool = False,
    **optimizer_kwargs,
) -> torch.optim.AdamW:
    """
    Create an optimizer for the given model.

        Modes:
        - Default (both flags False): apply provided weight_decay uniformly to all trainable params.
        - A-only: apply weight decay only to LoRA "A" matrices (names contain "lora_A" or
            "lora_embedding_A"); all other trainable params get weight_decay=0.0.
        - B-only: apply weight decay only to LoRA "B" matrices (names contain "lora_B" or
            "lora_embedding_B"); all other trainable params get weight_decay=0.0.
        - A+B: if both flags True, apply weight decay to both A and B matrices; others get 0.0.

    Args:
        model: The model with LoRA parameters.
        lora_a_weight_decay_only: If True, decay LoRA A params only (or with B if both True).
        lora_b_weight_decay_only: If True, decay LoRA B params only (or with A if both True).
        **optimizer_kwargs: Keyword args forwarded to AdamW. If any "only" flag is set, the
            global "weight_decay" is taken for LoRA groups and others get 0.0.

    Returns:
        An instantiated torch.optim.AdamW.
    """
    if lora_a_weight_decay_only or lora_b_weight_decay_only:
        wd = float(optimizer_kwargs.pop("weight_decay", 0.0))
        decay, no_decay = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            is_a = ("lora_A" in name) or ("lora_embedding_A" in name)
            is_b = ("lora_B" in name) or ("lora_embedding_B" in name)
            should_decay = (lora_a_weight_decay_only and is_a) or (lora_b_weight_decay_only and is_b)
            if should_decay:
                decay.append(p)
            else:
                no_decay.append(p)

        param_groups = [
            {"params": decay, "weight_decay": wd},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(param_groups, **optimizer_kwargs)

    # Default: standard optimizer over all trainable params
    params = (p for p in model.parameters() if p.requires_grad)
    return torch.optim.AdamW(params, **optimizer_kwargs)

def num_parameters(module: torch.nn.Module, requires_grad: bool | None = None) -> int:
    total = 0
    for p in module.parameters():
        if requires_grad is None or p.requires_grad == requires_grad:
            total += p.numel()
    return total

def iter_or_None(dataloader):
    return None if dataloader is None else iter(dataloader)

def move_adapter_to_parent(run_path, adapter_name):
    adapter_path = run_path / adapter_name
    if adapter_path.exists():
        # Move all files from adapter_path to run_path
        for f in adapter_path.iterdir():
            f.rename(run_path / f.name)

        # Remove the adapter directory
        adapter_path.rmdir()

def get_relative_path(data: list[Path]) -> list[Path]:
    # If training data is in DATA_PATH, we save paths relative to DATA_PATH
    dataset_path = DATA_PATH.resolve()
    relative_paths = []
    for path in data:
        file_path = path.resolve()
        try:
            relative_paths.append(file_path.relative_to(dataset_path))
        except ValueError:
            relative_paths.append(file_path)

    return relative_paths

def group_files(files: list[Path]):
    parent_files = [(str(f.parent), f.name) for f in files]
    grouped_files = {}
    for parent, name in parent_files:
        if parent not in grouped_files:
            grouped_files[parent] = []
        grouped_files[parent].append(name)

    return grouped_files

class InfiniteSampler(Sampler):
    def __init__(self, data_source_length):
        self.data_source_length = data_source_length

    def __iter__(self):
        while True:
            yield from torch.randperm(self.data_source_length)

    def __len__(self):
        return float('inf')

class GroupInfiniteSampler(Sampler):
    def __init__(self, data_source_length, dp_rank, seed=None):
        """
        Initializes the sampler with data source length and optional seeding for reproducibility.

        Args:
            data_source_length (int): The length of the data source or dataset.
            dp_rank (int): Identifier for the data parallel group. Samplers with the same group ID
                               will generate the same sequence of indices if the seed is provided.
            seed (int, optional): Optional seed for random number generation to ensure reproducibility
                                  across different runs but consistency within a DP group.
        """
        self.data_source_length = data_source_length
        self.dp_rank = dp_rank
        self.seed = seed
        self.generator = torch.Generator()  # Initialize generator

        # Seed the generator
        if self.seed is not None:
            self.generator.manual_seed(self.seed + self.dp_rank)
            random.seed(self.seed + self.dp_rank) # IMPORTANT, the code hangs without this line

        print(f'Init GroupInfiniteSampler with {seed=}, {dp_rank=}')

    def __iter__(self):
        while True:
            yield from torch.randperm(self.data_source_length, generator=self.generator)

    def __len__(self):
        return float('inf')

class NewGroupInfiniteSampler(Sampler):
    def __init__(self, weights, dp_rank=0, seed=None):
        """
        Infinite sampler that performs weighted sampling with replacement (multinomial).

        Args:
            weights (Tensor | list[float]): Per-sample weights; if provided, enables weighted mode.
            dp_rank (int): Data-parallel rank used to derive RNG seeds.
            seed (int | None): Base seed. Effective seed becomes (seed + dp_rank).
        """
        self.weights = weights
        self.dp_rank = dp_rank
        self.seed = seed
        self.generator = torch.Generator()

        if self.seed is not None:
            self.generator.manual_seed(self.seed + self.dp_rank)
            random.seed(self.seed + self.dp_rank)  # IMPORTANT: prevents hangs observed without it

    def __iter__(self):
        while True:
            idxs = torch.multinomial(
                self.weights,
                num_samples=len(self.weights),
                replacement=True,
                generator=self.generator,
            )
            yield from idxs.tolist()

def get_ip_address(hostname):
    """Resolve hostname to IP address."""
    return socket.gethostbyname(hostname)

@torch.no_grad()
def ema_update(model, alpha=0.99):
    model_state_dict = model.state_dict()
    for p_name, param in model.named_parameters():
        if "lora" in p_name and "default" in p_name:
            ema_name = p_name.replace("default", "ema")
            ema_param = model_state_dict[ema_name]
            ema_param.data._local_tensor.mul_(alpha).add_(param.data._local_tensor, alpha=1 - alpha)

def path_based_tags_assigner(step: STStepMessages) -> set[str]:
    tags = set()
    if step.metadata.get("path", None):
        # parse path to extract group
        path_parts = step.metadata["path"].split("/")
        if len(path_parts) > 2:
            group = path_parts[-2]
            tags.add(group)
    return tags

def assign_tags_from_step_labels(step: STStepMessages) -> set[str]:
    tags = set()
    if step_tag:=step.metadata.get("labels", []):
        tags.update(step_tag)
    return tags
