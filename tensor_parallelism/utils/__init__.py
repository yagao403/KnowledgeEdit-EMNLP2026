"""Utilities required by the paper's FSDP2 training path."""

import torch

from .helper import copy_directory_contents
from .load_utils import (
    get_nested_attribute,
    load_weight_from_state_dict_and_buffer_for_FSDP2,
    move_adapters_to_cpu_for_FSDP2,
    move_adapters_to_cpu_for_FSDP_TP,
)


def str_to_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if not isinstance(dtype, str):
        return dtype
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "f32":
        return torch.float32
    raise ValueError(f"Unknown dtype: {dtype}")
