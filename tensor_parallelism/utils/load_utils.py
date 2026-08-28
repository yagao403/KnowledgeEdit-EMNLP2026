import torch
import itertools
from typing import Dict, List, Tuple
import re
from tqdm import tqdm
from torch.distributed.fsdp._init_utils import _get_modules_to_materialize
from torch.distributed.fsdp import (FullyShardedDataParallel as FSDP,
                                    ShardedStateDictConfig,
                                    ShardedOptimStateDictConfig,
                                    )
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
from torch.distributed.checkpoint import FileSystemWriter, FileSystemReader, DefaultSavePlanner, DefaultLoadPlanner
import torch.distributed.checkpoint as dc
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor import DTensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.state_dict_loader import load as distributed_load
from torch.distributed.checkpoint.optimizer import load_sharded_optimizer_state_dict
from ..configs import fsdp_config
from torch.distributed.fsdp._unshard_param_utils import _unshard_params, _unshard_params_for_summon
from .log_utils import rank_print
import gc
from huggingface_hub import hf_hub_download, list_repo_files
from safetensors.torch import load_file
import time
from pathlib import Path
from typing import Union

def get_checkpoint_shards_filepaths(model_id):
    files = list_repo_files(model_id)
    safetensors_files = [f for f in files if f.endswith(".safetensors")]
    cached_paths = [hf_hub_download(model_id, filename) for filename in safetensors_files]
    return cached_paths

def get_nested_attribute(obj, attr_path) -> Union[torch.Tensor, nn.Module]:
    """
    IMPORTANT FUNCTION: used by both experimental runs and official training runs
    Recursively gets the nested attribute of an object.
    """
    attributes = attr_path.split('.')
    for attr in attributes:
        obj = getattr(obj, attr)
    return obj

def copy_to_nested_attribute(obj, attr_path, value, is_weight=True):
    """Copies value to the nested attribute of an object using the copy_ method."""
    attributes = attr_path.split('.')
    for attr in attributes[:-1]:
        obj = getattr(obj, attr)
    final_attr = getattr(obj, attributes[-1])
    # final_attr = get_nested_attribute(obj, attr_path)
    if is_weight:
        try:
            final_attr.data._local_tensor.copy_(value)
        except:
            final_attr.data.copy_(value)
    else:
        final_attr.copy_(value)

def load_from_state_dict_old(model, module_name, rank, tp_mesh, cpu_state_dict):
    module_state = cpu_state_dict[module_name]
    tp_local_rank = tp_mesh.get_local_rank()

    local_shape = get_nested_attribute(model, module_name)._local_tensor.shape
    module_shape = module_state.shape
    if len(local_shape) == 2:
        h, w = module_shape
        row_factor = module_shape[0] // local_shape[0] # The weight tensors are transposed in linears
        col_factor = module_shape[1] // local_shape[1]
        if col_factor > 1 and row_factor > 1:
            local_module_state = module_state[h//row_factor*tp_local_rank: h//row_factor*(tp_local_rank+1),
                                    w//col_factor*tp_local_rank: w//col_factor*(tp_local_rank+1)].to(rank)
        elif col_factor > 1:
            local_module_state = module_state[:, w//col_factor*tp_local_rank: w//col_factor*(tp_local_rank+1)].to(rank)
        elif row_factor > 1:
            local_module_state = module_state[h//row_factor*tp_local_rank: h//row_factor*(tp_local_rank+1), :].to(rank)
        else:
            local_module_state = module_state.to(rank)
    elif len(local_shape) == 1:
        h, = module_shape
        row_factor = module_shape[0] // local_shape[0]
        if row_factor > 1:
            local_module_state = module_state[h//row_factor*tp_local_rank: h//row_factor*(tp_local_rank+1)].to(rank)
        else:
            local_module_state = module_state.to(rank)

    copy_to_nested_attribute(model, module_name, local_module_state, is_weight=True)

def load_from_state_dict(model, module_name, rank, tp_mesh, module_state):
    # module_state = cpu_state_dict[module_name]
    tp_local_rank = tp_mesh.get_local_rank()
    tp_group = tp_mesh.get_group()
    module_shape = module_state.shape
    local_shape = get_nested_attribute(model, module_name)._local_tensor.shape
    weight_dtype = get_nested_attribute(model, module_name).data._local_tensor.dtype

    if tp_local_rank == 0:
        module_state = module_state.to(weight_dtype).to(rank)
    else:
        module_state = torch.empty(module_shape, dtype=weight_dtype).to(rank)

    module_state = module_state.contiguous()
    dist.broadcast(module_state, src=0, group=tp_group)

    if len(local_shape) == 2:
        h, w = module_shape
        row_factor = module_shape[0] // local_shape[0] # The weight tensors are transposed in linears
        col_factor = module_shape[1] // local_shape[1]
        if col_factor > 1 and row_factor > 1:
            local_module_state = module_state[h//row_factor*tp_local_rank: h//row_factor*(tp_local_rank+1),
                                    w//col_factor*tp_local_rank: w//col_factor*(tp_local_rank+1)]
        elif col_factor > 1:
            local_module_state = module_state[:, w//col_factor*tp_local_rank: w//col_factor*(tp_local_rank+1)]
        elif row_factor > 1:
            local_module_state = module_state[h//row_factor*tp_local_rank: h//row_factor*(tp_local_rank+1), :]
        else:
            local_module_state = module_state
    elif len(local_shape) == 1:
        h, = module_shape
        row_factor = module_shape[0] // local_shape[0]
        if row_factor > 1:
            local_module_state = module_state[h//row_factor*tp_local_rank: h//row_factor*(tp_local_rank+1)]
        else:
            local_module_state = module_state

    copy_to_nested_attribute(model, module_name, local_module_state, is_weight=True)

def get_decoder_block_path(module_path):
    path_components = []
    is_decoder_block = False
    for component in module_path.split('.'):
        path_components.append(component)
        # check if component is a number with regex
        if re.match(r'^\d+$', component):
            is_decoder_block = True
            break

    rest_of_path = [component for component in module_path.split('.') if component not in path_components]

    if is_decoder_block:
        return '.'.join(path_components), '.'.join(rest_of_path)
    else:
        return None, None

def load_weight_from_state_dict_and_buffer_for_FSDP(
    rank,
    *,
    device_mesh: DeviceMesh,
    model: nn.Module,
    cpu_state_dict: Dict,
    cpu_buffers: Dict,
):
    print("Loading weights from cpu_state_dict ...")
    module_list = [str(module_name) for module_name in cpu_state_dict]
    # sort the module list to make sure the order is the same across all ranks
    module_list = sorted(module_list)

    for module_name in tqdm(module_list):
        load_from_state_dict_for_FSDP(
            rank,
            model=model,
            module_name=module_name,
            device_mesh=device_mesh,
            module_state=cpu_state_dict[module_name]
        )

    print('Loading buffers ...')
    buffer_list = [buffer_name for buffer_name in cpu_buffers]
    buffer_list = sorted(buffer_list)
    for buffer_name in tqdm(buffer_list):
        load_from_buffer_for_FSDP(
            rank,
            model=model,
            buffer_name=buffer_name,
            device_mesh=device_mesh,
            cpu_buffer=cpu_buffers[buffer_name],
        )

def load_from_state_dict_for_FSDP(
    rank,
    *,
    model: nn.Module,
    module_name: str,
    device_mesh: DeviceMesh,
    module_state,
):
    dp_mesh = device_mesh
    dp_local_rank = dp_mesh.get_local_rank()
    dp_group = dp_mesh.get_group()

    decoder_block_path, rest_of_path = get_decoder_block_path(module_name)

    if decoder_block_path is not None:
        module = get_nested_attribute(model, decoder_block_path)
    else:
        module = model
        rest_of_path = module_name

    module_dtype = get_nested_attribute(module, rest_of_path).dtype

    # Create empty tensor for non-zero ranks
    if dp_local_rank == 0:
        module_state = module_state.to(module_dtype).to(rank)
    else:
        module_state = torch.empty(module_state.shape, dtype=module_dtype).to(rank)

    # Broadcast the local_module_state to other ranks in the same dp group
    module_state = module_state.contiguous()
    dist.broadcast(module_state, src=0, group=dp_group)

    with _unshard_params(module,
                            recurse=False,
                            writeback=True,
                            rank0_only=False,
                            offload_to_cpu=False,
                            with_grads=False,
                        ):
        weight = get_nested_attribute(module, rest_of_path)
        weight.data.copy_(module_state)

def load_from_state_dict_for_FSDP_TP(model, module_name, rank, device_mesh, module_state):
    # print(f'Rank {rank} Loading {module_name=}')
    # print("Preparing local module state ...")

    tp_mesh = device_mesh['tp']
    dp_mesh = device_mesh['dp']
    tp_local_rank = tp_mesh.get_local_rank()
    dp_local_rank = dp_mesh.get_local_rank()
    dp_group = device_mesh.get_group(mesh_dim="dp")
    # create a group of ranks that have same tp_local_rank but different dp_local_rank
    # vertical_subgroup = dist.new_group(ranks=[i*tp_mesh.size() + tp_local_rank for i in range(dp_mesh.size())])
    # print(f'{rank = } {tp_local_rank = } {dp_local_rank = } {dp_group.size() = } {dp_group.rank() = }')

    decoder_block_path, rest_of_path = get_decoder_block_path(module_name)
    # print(f'{decoder_block_path=} {rest_of_path=}')

    if decoder_block_path is not None:
        module = get_nested_attribute(model, decoder_block_path)
    else:
        module = model
        rest_of_path = module_name

    with _unshard_params(
        module,
        recurse=False,
        writeback=True,
        rank0_only=False,
        offload_to_cpu=False,
        with_grads=False,
    ):
        weight = get_nested_attribute(module, rest_of_path)
        # print(f'{rank = } {weight.data.shape=} {type(weight.data)=}')
        local_shape = weight.data._local_tensor.shape
        weight_dtype = weight.data._local_tensor.dtype

    module_shape = module_state.shape

    # print(f'{rank =} {module_name=} {local_shape=} {module_shape=}')

    # if decoder_block_path is None or dp_local_rank == 0:
    if dp_local_rank == 0:
        # Make a deep copy of module_state to avoid modifying the original state
        module_state = module_state.to(weight_dtype)
        # print(f'{rank=} reached here {len(local_shape)=} {module_shape=}')
        if len(local_shape) == 2:
            h, w = module_shape
            row_factor = module_shape[0] // local_shape[0] # The weight tensors are transposed in linears
            col_factor = module_shape[1] // local_shape[1]
            if col_factor > 1 and row_factor > 1:
                local_module_state = module_state[h//row_factor*tp_local_rank: h//row_factor*(tp_local_rank+1),
                                        w//col_factor*tp_local_rank: w//col_factor*(tp_local_rank+1)].to(rank)
            elif col_factor > 1:
                local_module_state = module_state[:, w//col_factor*tp_local_rank: w//col_factor*(tp_local_rank+1)].to(rank)
            elif row_factor > 1:
                local_module_state = module_state[h//row_factor*tp_local_rank: h//row_factor*(tp_local_rank+1), :].to(rank)
            else:
                local_module_state = module_state.to(rank)
        elif len(local_shape) == 1:
            h, = module_shape
            row_factor = module_shape[0] // local_shape[0]
            if row_factor > 1:
                local_module_state = module_state[h//row_factor*tp_local_rank: h//row_factor*(tp_local_rank+1)].to(rank)
            else:
                # print(f'{rank=} reached here 2 {module_name=} {module_state=}')
                local_module_state = module_state.to(rank)
                # print(f'{rank=} reached here 3')
        # print(f'{module_name=} initialized with real data {local_module_state.shape = } at rank {rank=}')
    else: # Create empty tensor for non-zero ranks
        if len(local_shape) == 2:
            h, w = module_shape
            row_factor = module_shape[0] // local_shape[0]
            col_factor = module_shape[1] // local_shape[1]
            if col_factor > 1 and row_factor > 1:
                local_module_state = torch.empty(h//row_factor, w//col_factor, dtype=weight_dtype).to(rank)
            elif col_factor > 1:
                local_module_state = torch.empty(h, w//col_factor, dtype=weight_dtype).to(rank)
            elif row_factor > 1:
                local_module_state = torch.empty(h//row_factor, w, dtype=weight_dtype).to(rank)
            else:
                local_module_state = torch.empty(h, w, dtype=weight_dtype).to(rank)
        elif len(local_shape) == 1:
            h, = module_shape
            row_factor = module_shape[0] // local_shape[0]
            if row_factor > 1:
                local_module_state = torch.empty(h//row_factor, dtype=weight_dtype).to(rank)
            else:
                local_module_state = torch.empty(h, dtype=weight_dtype).to(rank)
        # print(f'{module_name=} initialized with empty tensor at rank {rank=}')

    # print(f'{local_module_state.shape = } for {module_name=} at rank {rank=}')
    # dist.barrier()
    # if decoder_block_path is not None:
    if True:
        # send the local_module_state to other ranks in the same dp group
        # print(f'{rank=} waiting for local_module_state from {tp_local_rank=}')
        local_module_state = local_module_state.contiguous()
        dist.broadcast(local_module_state, src=tp_local_rank, group=dp_group)

    with _unshard_params(module,
                            recurse=False,
                            writeback=True,
                            rank0_only=False,
                            offload_to_cpu=False,
                            with_grads=False,
                        ):
        weight = get_nested_attribute(module, rest_of_path)
        weight.data._local_tensor.copy_(local_module_state)
        # print(f'data of {module_name=} copied to {rank=} {tp_local_rank=}')

    # dist.barrier()
    # print(f'Rank {rank} finished loading {module_name=}')

def load_from_buffer_for_FSDP(
    rank,
    *,
    model: nn.Module,
    buffer_name: str,
    device_mesh: DeviceMesh,
    cpu_buffer,
):
    dp_mesh = device_mesh
    dp_local_rank = dp_mesh.get_local_rank()
    dp_group = dp_mesh.get_group()

    buffer = get_nested_attribute(model, buffer_name)
    local_shape = buffer.shape
    buffer_dtype = buffer.dtype
    module_shape = cpu_buffer.shape

    # Create empty tensor for no-zero ranks
    if dp_local_rank == 0:
        local_buffer = cpu_buffer.to(buffer_dtype).to(rank)
    else:
        local_buffer = torch.empty(module_shape, dtype=buffer_dtype).to(rank)

    # Broadcast the local_module_state to other ranks in the same dp group
    local_buffer = local_buffer.contiguous()
    dist.broadcast(local_buffer, src=0, group=dp_group)

    buffer.copy_(local_buffer)

def load_from_buffer(model, module_name, rank, tp_mesh, module_buffer):
    tp_local_rank = tp_mesh.get_local_rank()
    tp_group = tp_mesh.get_group()
    local_shape = get_nested_attribute(model, module_name).shape
    module_shape = module_buffer.shape
    weight_dtype = get_nested_attribute(model, module_name).dtype

    if tp_local_rank == 0:
        module_buffer = module_buffer.to(weight_dtype).to(rank)
    else:
        module_buffer = torch.empty(module_shape, dtype=weight_dtype).to(rank)

    module_buffer = module_buffer.contiguous()
    dist.broadcast(module_buffer, src=0, group=tp_group)

    if len(local_shape) == 2:
        h, w = module_shape
        row_factor = module_shape[0] // local_shape[0] # The weight tensors are transposed in linears
        col_factor = module_shape[1] // local_shape[1]
        if col_factor > 1 and row_factor > 1:
            local_module_state = module_buffer[h//row_factor*tp_local_rank: h//row_factor*(tp_local_rank+1),
                                    w//col_factor*tp_local_rank: w//col_factor*(tp_local_rank+1)].to(rank)
        elif col_factor > 1:
            local_module_state = module_buffer[:, w//col_factor*tp_local_rank: w//col_factor*(tp_local_rank+1)].to(rank)
        elif row_factor > 1:
            local_module_state = module_buffer[h//row_factor*tp_local_rank: h//row_factor*(tp_local_rank+1), :].to(rank)
        else:
            local_module_state = module_buffer.to(rank)
    elif len(local_shape) == 1:
        h, = module_shape
        row_factor = module_shape[0] // local_shape[0]
        if row_factor > 1:
            local_module_state = module_buffer[h//row_factor*tp_local_rank: h//row_factor*(tp_local_rank+1)].to(rank)
        else:
            local_module_state = module_buffer.to(rank)

    copy_to_nested_attribute(model, module_name, local_module_state, is_weight=False)

def load_from_buffer_for_FSDP_TP(model, module_name, rank, device_mesh, module_buffer):
    tp_mesh = device_mesh['tp']
    dp_mesh = device_mesh['dp']

    tp_local_rank = tp_mesh.get_local_rank()
    dp_local_rank = dp_mesh.get_local_rank()
    dp_group = device_mesh.get_group(mesh_dim="dp")
    local_shape = get_nested_attribute(model, module_name).shape
    module_shape = module_buffer.shape
    if dp_local_rank == 0:
        if len(local_shape) == 2:
            h, w = module_shape
            row_factor = module_shape[0] // local_shape[0] # The weight tensors are transposed in linears
            col_factor = module_shape[1] // local_shape[1]
            if col_factor > 1 and row_factor > 1:
                local_module_state = module_buffer[h//row_factor*tp_local_rank: h//row_factor*(tp_local_rank+1),
                                        w//col_factor*tp_local_rank: w//col_factor*(tp_local_rank+1)].to(rank)
            elif col_factor > 1:
                local_module_state = module_buffer[:, w//col_factor*tp_local_rank: w//col_factor*(tp_local_rank+1)].to(rank)
            elif row_factor > 1:
                local_module_state = module_buffer[h//row_factor*tp_local_rank: h//row_factor*(tp_local_rank+1), :].to(rank)
            else:
                local_module_state = module_buffer.to(rank)
        elif len(local_shape) == 1:
            h, = module_shape
            row_factor = module_shape[0] // local_shape[0]
            if row_factor > 1:
                local_module_state = module_buffer[h//row_factor*tp_local_rank: h//row_factor*(tp_local_rank+1)].to(rank)
            else:
                local_module_state = module_buffer.to(rank)
    else: # Create empty tensor for non-zero ranks
        if len(local_shape) == 2:
            h, w = module_shape
            row_factor = module_shape[0] // local_shape[0]
            col_factor = module_shape[1] // local_shape[1]
            if col_factor > 1 and row_factor > 1:
                local_module_state = torch.empty(h//row_factor, w//col_factor).to(rank)
            elif col_factor > 1:
                local_module_state = torch.empty(h, w//col_factor).to(rank)
            elif row_factor > 1:
                local_module_state = torch.empty(h//row_factor, w).to(rank)
            else:
                local_module_state = torch.empty(h, w).to(rank)
        elif len(local_shape) == 1:
            h, = module_shape
            row_factor = module_shape[0] // local_shape[0]
            if row_factor > 1:
                local_module_state = torch.empty(h//row_factor).to(rank)
            else:
                local_module_state = torch.empty(h).to(rank)

    # send the local_module_state to other ranks in the same dp group
    local_module_state = local_module_state.contiguous()
    dist.broadcast(local_module_state, src=tp_local_rank, group=dp_group)

    copy_to_nested_attribute(model, module_name, local_module_state, is_weight=False)

def load_weight_from_state_dict_and_buffer(model, cpu_state_dict, cpu_buffers, rank, tp_mesh):
    print("Loading weights from cpu_state_dict ...")
    module_list = [str(module_name) for module_name in cpu_state_dict]
    # sort the module list to make sure the order is the same across all ranks
    module_list = sorted(module_list)

    for module_name in tqdm(module_list):
        load_from_state_dict(model,
                            module_name,
                            rank,
                            tp_mesh,
                            cpu_state_dict[module_name])

    print('Loading buffers ...')
    buffer_list = [buffer_name for buffer_name in cpu_buffers]
    buffer_list = sorted(buffer_list)

    for buffer_name in tqdm(buffer_list):
        load_from_buffer(model,
                        buffer_name,
                        rank,
                        tp_mesh,
                        cpu_buffers[buffer_name])

    print(f"TP Sharded Model device: {model.device=}")

def load_weight_from_state_dict_and_buffer_for_FSDP_TP(model, cpu_state_dict, cpu_buffers, rank, device_mesh):
    '''
    IMPORTANT FUNCTION: used by both experimental runs and official training runs
    '''
    print("Loading weights from cpu_state_dict ...")
    module_list = [str(module_name) for module_name in cpu_state_dict]
    # sort the module list to make sure the order is the same across all ranks
    module_list = sorted(module_list)

    for module_name in tqdm(module_list):
        load_from_state_dict_for_FSDP_TP(model,
                            module_name,
                            rank,
                            device_mesh,
                            cpu_state_dict[module_name],
                            )

    print('Loading buffers ...')
    buffer_list = [buffer_name for buffer_name in cpu_buffers]
    buffer_list = sorted(buffer_list)
    for buffer_name in tqdm(buffer_list):
        load_from_buffer_for_FSDP_TP(model,
                        buffer_name,
                        rank,
                        device_mesh,
                        cpu_buffers[buffer_name],
                        )

    print(f"FSDP-TP Sharded Model device: {model.device=}")

def load_from_state_dict_for_FSDP2(
    rank,
    *,
    model: nn.Module,
    module_name: str,
    device_mesh: DeviceMesh,
    module_state,
):
    # assert if device_mesh dimension is 1D
    assert device_mesh.ndim == 1, f"Device mesh should be 1D, but got {device_mesh.ndim=}"
    dp_mesh = device_mesh
    dp_size = dp_mesh.size()
    dp_local_rank = dp_mesh.get_local_rank()

    decoder_block_path, rest_of_path = get_decoder_block_path(module_name)
    if decoder_block_path is not None:
        module = get_nested_attribute(model, decoder_block_path)
    else:
        module = model
        rest_of_path = module_name

    weight = get_nested_attribute(module, rest_of_path)
    local_shape = weight.data._local_tensor.shape
    weight_dtype = weight.data._local_tensor.dtype
    # print(f'{rank = } {module_name=} {local_shape=} {weight_dtype=}')

    module_shape = module_state.shape

    # Convert the cpu module_state to weight_dtype
    module_state = module_state.to(weight_dtype)
    if len(local_shape) >= 2:
        h = module_shape[0]
        row_factor = module_shape[0] // local_shape[0] # The weight tensors are transposed in linears
        col_factor = module_shape[1] // local_shape[1]

        # NOTE: For FSDP2-only, it always shards the row dimension and it's always a 1D sharding.
        assert col_factor == 1, f"FSDP2 only supports 1D sharding on rows, but receive {col_factor=}"
        assert row_factor == dp_size, f"Row factor {row_factor} is not equal to dp_size {dp_size} for {module_name=}"
        local_module_state = module_state[h//row_factor*dp_local_rank : h//row_factor*(dp_local_rank+1)].to(rank)

    elif len(local_shape) == 1:
        h = module_shape[0]
        row_factor = module_shape[0] // local_shape[0]

        assert row_factor == dp_size, f"Row factor {row_factor} is not equal to dp_size {dp_size} for {module_name=}"
        local_module_state = module_state[h//row_factor*dp_local_rank : h//row_factor*(dp_local_rank+1)].to(rank)

    else:
        raise ValueError(f"Unsupported module shape {module_shape} for {module_name=}")

    with torch.no_grad():
        weight.data._local_tensor.copy_(local_module_state)

def load_from_buffer_for_FSDP2(
        rank,
        *,
        model: nn.Module,
        module_name: str,
        device_mesh: DeviceMesh,
        module_buffer,
    ):
    # assert if device_mesh dimension is 1D
    assert device_mesh.ndim == 1, f"Device mesh should be 1D, but got {device_mesh.ndim=}"
    dp_mesh = device_mesh
    dp_group = device_mesh.get_group()
    buffer = get_nested_attribute(model, module_name)
    buffer_dtype = buffer.dtype
    module_shape = module_buffer.shape
    # Check if buffer is DTensor, raise error
    assert not isinstance(buffer, DTensor), f"Buffer {module_name} is a DTensor, which is not supported yet"

    local_buffer_state = module_buffer.to(buffer_dtype).to(rank)

    with torch.no_grad():
        buffer.copy_(local_buffer_state)

def load_weight_from_state_dict_and_buffer_for_FSDP2(
        rank,
        *,
        device_mesh: DeviceMesh,
        model: nn.Module,
        cpu_state_dict: Dict,
        cpu_buffers: Dict,
    ):
    print("Loading weights from cpu_state_dict ...")
    module_list = [str(module_name) for module_name in cpu_state_dict]
    # sort the module list to make sure the order is the same across all ranks
    module_list = sorted(module_list)

    for module_name in tqdm(module_list):
        load_from_state_dict_for_FSDP2(
            rank,
            model=model,
            module_name=module_name,
            device_mesh=device_mesh,
            module_state=cpu_state_dict[module_name],
        )

    print('Loading buffers ...')
    buffer_list = [buffer_name for buffer_name in cpu_buffers]
    buffer_list = sorted(buffer_list)
    for buffer_name in tqdm(buffer_list):
        load_from_buffer_for_FSDP2(
            rank,
            model=model,
            module_name=buffer_name,
            device_mesh=device_mesh,
            module_buffer=cpu_buffers[buffer_name],
        )
    print(f"FSDP2 Sharded Model device: {model.device=}")

def load_from_state_dict_for_FSDP2_TP(
    rank,
    *,
    model: nn.Module,
    module_name: str,
    device_mesh: DeviceMesh,
    module_state,
):
    tp_mesh = device_mesh['tp']
    dp_mesh = device_mesh['dp']
    tp_size = tp_mesh.size()
    dp_size = dp_mesh.size()
    tp_local_rank = tp_mesh.get_local_rank()
    dp_local_rank = dp_mesh.get_local_rank()
    dp_group = device_mesh.get_group(mesh_dim="dp")

    decoder_block_path, rest_of_path = get_decoder_block_path(module_name)
    if decoder_block_path is not None:
        module = get_nested_attribute(model, decoder_block_path)
    else:
        module = model
        rest_of_path = module_name

    weight = get_nested_attribute(module, rest_of_path)
    global_shape = weight.shape
    local_shape = weight.data._local_tensor.shape
    weight_dtype = weight.data._local_tensor.dtype

    module_shape = module_state.shape

    # Make a deep copy of module_state to avoid modifying the original state
    module_state = module_state.to(weight_dtype)
    if len(local_shape) == 2:
        h, w = module_shape
        row_factor = module_shape[0] // local_shape[0] # The weight tensors are transposed in linears
        col_factor = module_shape[1] // local_shape[1]

        # NOTE: Here, `row_factor` actually refers to the *column-wise* TP parallelism factor,
        # and `col_factor` refers to the *row-wise* TP parallelism factor.
        # So, if a tensor is column-wise parallelized, i.e. the actual weight is sharded on
        # row dimension, FSDP2 will further shard it along the row dimension (since FSDP2
        # always shards along the row dimension). This results in a 1D sharded tensor.
        # For example, a column-wise TP parallelized `lm_head` weight will end up with a placement like:
        # (_StridedShard(dim=0, sf=<tp_size>), Shard(dim=0)).
        # On the other hand, if a tensor is row-wise sharded (i.e., sharded along the column dimension),
        # combining that with FSDP2’s row-wise sharding results in a 2D sharded tensor.
        if col_factor > 1 and row_factor > 1:
            local_module_state = module_state[h//row_factor*dp_local_rank : h//row_factor*(dp_local_rank+1),
                                    w//col_factor*tp_local_rank : w//col_factor*(tp_local_rank+1)].to(rank)
        elif col_factor == 1 and row_factor > 1:
            if row_factor == dp_size: # Replicate across TP
                local_module_state = module_state[h//row_factor*dp_local_rank : h//row_factor*(dp_local_rank+1), :].to(rank)
            elif row_factor > dp_size: # 1D strided sharding
                local_module_state = torch.empty(h//row_factor, w, dtype=weight_dtype).to(rank)
                row_id = tp_local_rank * dp_size + dp_local_rank
                local_module_state = module_state[h//row_factor*row_id : h//row_factor*(row_id+1), :].to(rank)
            else:
                raise ValueError(f"Invalid row_factor {row_factor} for {module_name=}")
        # row_factor == 1, i.e., no FSDP2 sharded, which is weird but still valid
        else:
            raise ValueError(f"Invalid row_factor {row_factor} and col_factor {col_factor} for {module_name=}")
    elif len(local_shape) == 1:
        h, = module_shape
        row_factor = module_shape[0] // local_shape[0]
        if 'norm' in module_name: # norm layers are not TP sharded
            if row_factor == dp_size:
                local_module_state = module_state[h//row_factor*dp_local_rank : h//row_factor*(dp_local_rank+1)].to(rank)
            else:
                raise ValueError(f"Norm layer sharded detected but receive invalid row_factor {row_factor} for {module_name=}")
        else:
            raise ValueError(f"Assumed no FSDP2 sharded but still receive invalid local_shape {local_shape} global shape {global_shape} for {module_name=}")

    with torch.no_grad():
        weight.data._local_tensor.copy_(local_module_state)

def load_from_buffer_for_FSDP2_TP(
        rank,
        *,
        model: nn.Module,
        module_name: str,
        device_mesh: DeviceMesh,
        module_buffer,
    ):
    tp_mesh = device_mesh['tp']
    dp_mesh = device_mesh['dp']
    tp_local_rank = tp_mesh.get_local_rank()
    dp_local_rank = dp_mesh.get_local_rank()
    dp_group = device_mesh.get_group(mesh_dim="dp")
    buffer = get_nested_attribute(model, module_name)
    print(f'{type(buffer)=} {buffer.shape=} {buffer.dtype=} {module_name=}')
    buffer_dtype = buffer.dtype
    module_shape = module_buffer.shape
    # check if buffer is DTensor, raise error
    if isinstance(buffer, DTensor):
        raise ValueError(f"Buffer {module_name} is a DTensor, which is not supported yet")

    local_buffer_state = module_buffer.to(buffer_dtype).to(rank)

    with torch.no_grad():
        buffer.copy_(local_buffer_state)

def load_weight_from_state_dict_and_buffer_for_FSDP2_TP(
        rank,
        *,
        device_mesh: DeviceMesh,
        model: nn.Module,
        cpu_state_dict: Dict,
        cpu_buffers: Dict,
    ):
    print("Loading weights from cpu_state_dict ...")
    module_list = [str(module_name) for module_name in cpu_state_dict]
    # sort the module list to make sure the order is the same across all ranks
    module_list = sorted(module_list)

    for module_name in tqdm(module_list):
        load_from_state_dict_for_FSDP2_TP(
            rank,
            model=model,
            module_name=module_name,
            device_mesh=device_mesh,
            module_state=cpu_state_dict[module_name],
        )

    print('Loading buffers ...')
    buffer_list = [buffer_name for buffer_name in cpu_buffers]
    buffer_list = sorted(buffer_list)
    for buffer_name in tqdm(buffer_list):
        load_from_buffer_for_FSDP2_TP(
            rank,
            model=model,
            module_name=buffer_name,
            device_mesh=device_mesh,
            module_buffer=cpu_buffers[buffer_name],
        )
    print(f"FSDP2-TP Sharded Model device: {model.device=}")

def materialize_weights(model, rank, reset_param=False):
    modules_to_materialize = _get_modules_to_materialize(model, set())
    for module in tqdm(modules_to_materialize):
        # As a contract to the user, only call `reset_parameters()` if
        # the module has directly managed parameters/buffers
        module_state_iter = itertools.chain(
            module.parameters(recurse=False), module.buffers(recurse=False)
        )
        has_module_states = len(list(module_state_iter)) > 0
        if has_module_states:
            module.to_empty(device=rank, recurse=False)
            if reset_param:
                module.reset_parameters() # type: ignore[operator]

def save_distributed_state_dict_and_buffer(model, cpu_buffers, checkpoint_path, buffer_path):
    print('Saving into checkpoints ... ')
    model_state_dict = model.state_dict()
    fs_storage_writer = FileSystemWriter(checkpoint_path)
    dc.save(
        state_dict=model_state_dict,
        storage_writer=fs_storage_writer,
        process_group=dist.group.WORLD,
    )
    print(f'Saved succesfully at {checkpoint_path=}')
    if cpu_buffers is not None:
        print('Saving buffers ...')
        torch.save(cpu_buffers, buffer_path)
        print(f'Saved buffers at {buffer_path=}')

def save_distributed_state_dict_and_buffer_for_FSDP(model, cpu_buffers, checkpoint_path, buffer_path):
    print('Saving into checkpoints ... ')
    with FSDP.state_dict_type(
        model,
        state_dict_type=StateDictType.SHARDED_STATE_DICT,
        state_dict_config=ShardedStateDictConfig(offload_to_cpu=True),
        # optim_state_dict_config = FullOptimStateDictConfig(offload_to_cpu=True,),
    ):
        model_state_dict = {
            'model': model.state_dict()
        }
        fs_storage_writer = FileSystemWriter(checkpoint_path)
        dc.save(
            state_dict=model_state_dict,
            storage_writer=fs_storage_writer,
            process_group=dist.group.WORLD,
        )
        print(f'Saved succesfully at {checkpoint_path=}')

    if cpu_buffers is not None:
        print('Saving buffers ...')
        torch.save(cpu_buffers, buffer_path)
        print(f'Saved buffers at {buffer_path=}')

def save_distributed_state_dict_optimizer_and_buffer_for_FSDP(model,
                                                              cpu_buffers,
                                                              checkpoint_path,
                                                              buffer_path,
                                                              optimizer,
                                                              ):
    print('Saving state dict, optimizer and buffers into checkpoints ... ')
    with FSDP.state_dict_type(
        model,
        state_dict_type=StateDictType.SHARDED_STATE_DICT,
        state_dict_config=ShardedStateDictConfig(offload_to_cpu=True),
        optim_state_dict_config = ShardedOptimStateDictConfig(offload_to_cpu=True,),
    ):
        state_dict = {
            'model': model.state_dict(),
            'optimizer': FSDP.optim_state_dict(model, optimizer),
        }
        fs_storage_writer = FileSystemWriter(checkpoint_path)
        dc.save(
            state_dict=state_dict,
            storage_writer=fs_storage_writer,
            process_group=dist.group.WORLD,
        )

    if cpu_buffers is not None:
        print('Saving buffers ...')
        torch.save(cpu_buffers, buffer_path)
        print(f'Saved buffers at {buffer_path=}')

def load_distributed_state_dict_and_buffer(model, checkpoint_path, buffer_path, rank, device_mesh):
    print(f'Loading state dict and buffers from checkpoints {checkpoint_path} ... ')
    model_state_dict = model.state_dict()
    fs_storage_reader = FileSystemReader(checkpoint_path)
    distributed_load(
        state_dict=model_state_dict,
        storage_reader=fs_storage_reader,
        process_group=dist.group.WORLD,
    )
    model.load_state_dict(model_state_dict)
    print('State dict loaded. Loading buffers ...')
    cpu_buffers = torch.load(buffer_path)
    for buffer_name in cpu_buffers:
        load_from_buffer_for_FSDP_TP(model,
                        buffer_name,
                        rank,
                        device_mesh,
                        cpu_buffers[buffer_name])
    print(f'Buffers loaded on {rank=}.')

def load_distributed_state_dict_and_buffer_for_FSDP(model, checkpoint_path, buffer_path, rank, device_mesh):
    print(f'Loading state dict and buffers from checkpoints {checkpoint_path} ... ')
    with FSDP.state_dict_type(
        model,
        state_dict_type=StateDictType.SHARDED_STATE_DICT,
        state_dict_config=ShardedStateDictConfig(offload_to_cpu=True),
        # optim_state_dict_config = FullOptimStateDictConfig(offload_to_cpu=True,),
    ):
        model_state_dict = {
            'model': model.state_dict()
        }
        fs_storage_reader = FileSystemReader(checkpoint_path)
        distributed_load(
            state_dict=model_state_dict,
            storage_reader=fs_storage_reader,
            process_group=dist.group.WORLD,
        )
        model.load_state_dict(model_state_dict['model'])

    print('State dict loaded. Loading buffers ...')
    cpu_buffers = torch.load(buffer_path)
    for buffer_name in cpu_buffers:
        load_from_buffer_for_FSDP_TP(model,
                        buffer_name,
                        rank,
                        device_mesh,
                        cpu_buffers[buffer_name])
    print(f'Buffers loaded on {rank=}.')

def load_distributed_state_dict_optimizer_and_buffer_for_FSDP(model,
                                                              optimizer,
                                                              checkpoint_path,
                                                              buffer_path,
                                                              rank,
                                                              device_mesh):
    print(f'Loading state dict, optimizer and buffers from checkpoints {checkpoint_path} ... ')
    with FSDP.state_dict_type(
        model,
        state_dict_type=StateDictType.SHARDED_STATE_DICT,
        state_dict_config=ShardedStateDictConfig(offload_to_cpu=True),
        optim_state_dict_config = ShardedOptimStateDictConfig(offload_to_cpu=True,),
    ):
        module_state = {
            'model': model.state_dict()
        }
        fs_storage_reader = FileSystemReader(checkpoint_path)
        # print(f"{state_dict['optimizer']=}")
        dc.load(
            state_dict=module_state,
            storage_reader=fs_storage_reader,
            process_group=dist.group.WORLD,
        )
        model.load_state_dict(module_state['model'])
        print('Model loaded. Loading optimizer state dict ...')
        fs_storage_reader = FileSystemReader(checkpoint_path)
        optim_state = load_sharded_optimizer_state_dict(
                            model_state_dict=module_state['model'],
                            optimizer_key='optimizer',
                            storage_reader=fs_storage_reader,
                        )

        flattened_osd = FSDP.optim_state_dict_to_load(
            optim_state_dict=optim_state['optimizer'],
            model=model,
            optim=optimizer,
        )
        optimizer.load_state_dict(flattened_osd)
        print('Optimizer state dict loaded. Loading buffers ...')

    cpu_buffers = torch.load(buffer_path)
    for buffer_name in cpu_buffers:
        load_from_buffer_for_FSDP_TP(model,
                        buffer_name,
                        rank,
                        device_mesh,
                        cpu_buffers[buffer_name])
    print(f'Buffers loaded on {rank=}.')

def move_adapters_to_cpu_for_FSDP_TP(rank,
                                    model,
                                    cpu_state_dict,
                                    device_mesh
                                    ):
    '''
    IMPORTANT FUNCTION: used by official training runs
    '''
    rank_print(rank, 'Moving adapters to cpu')
    peft_weights = [weight_name for weight_name in cpu_state_dict if 'lora' in weight_name]
    tp_rank = device_mesh['tp'].get_local_rank() if 'tp' in device_mesh.mesh_dim_names else 0
    dp_rank = device_mesh['dp'].get_local_rank()
    global_rank = rank
    tp_size = device_mesh['tp'].size() if 'tp' in device_mesh.mesh_dim_names else 1

    for peft_weight in tqdm(peft_weights):
        cpu_module_state = cpu_state_dict[peft_weight]
        module_shape = cpu_module_state.shape
        # rank_print(rank, f'Before: {cpu_module_state=}')

        decoder_block_path, rest_of_path = get_decoder_block_path(peft_weight)

        if decoder_block_path is not None:
            module = get_nested_attribute(model, decoder_block_path)
        else:
            module = model
            rest_of_path = peft_weight

        # rank_print(rank, f'{rank=} {peft_weight=} before: {cpu_state_dict[peft_weight]=}')
        with _unshard_params(module,
                            recurse=False,
                            writeback=False,
                            rank0_only=False,
                            offload_to_cpu=False,
                            with_grads=False,):
            weight = get_nested_attribute(module, rest_of_path)
            local_weight = weight.data._local_tensor
            local_shape = local_weight.shape

        sharded_layout = None
        sharded_dims = {
            'dp_dim': None,
            'tp_dim': None,
        }
        if len(local_shape) == 2:
            h, w = module_shape
            row_factor = module_shape[0] // local_shape[0]
            col_factor = module_shape[1] // local_shape[1]
            if row_factor == 1 and col_factor == 1:
                sharded_dims['dp_dim'] = None
                sharded_dims['tp_dim'] = None
                sharded_layout = 'REPLICATED'
            elif row_factor == tp_size and col_factor == 1:
                sharded_dims['dp_dim'] = None
                sharded_dims['tp_dim'] = 0
                sharded_layout = 'COLWISE'
            elif row_factor == 1 and col_factor == tp_size:
                sharded_dims['dp_dim'] = None
                sharded_dims['tp_dim'] = 1
                sharded_layout = 'ROWWISE'
            else:
                raise AssertionError('Invalid sharded weight detected')
        elif local_shape == 1:
            h, = module_shape
            row_factor = module_shape[0] // local_shape[0]
            if row_factor == 1:
                sharded_dims['dp_dim'] = None
                sharded_dims['tp_dim'] = None
                sharded_layout = 'REPLICATED'
            else:
                raise AssertionError('Invalid sharded weight detected')
        else:
            raise AssertionError('Empty local tensor detected')

        def gather_weights(rank, local_weight, dim, process_group):
            if rank == 0:
                # List to collect all sharded weights from all devices
                gathered_weights = [torch.zeros_like(local_weight) for _ in range(dist.get_world_size(group=process_group))]
                # Gathering weights across all devices
                dist.gather(local_weight, gather_list=gathered_weights, group=process_group)
            else:
                dist.gather(local_weight, dst=0, group=process_group)

            if rank == 0:
                # Concatenate weights along the appropriate dimension if necessary
                full_weight = torch.cat(gathered_weights, dim=dim)
                # print(f'Full weight shape: {full_weight.shape} {full_weight=}')
                return full_weight

        with _unshard_params(module,
                            recurse=False,
                            writeback=False,
                            rank0_only=False,
                            offload_to_cpu=False,
                            with_grads=False,):
            weight = get_nested_attribute(module, rest_of_path)
            local_weight = weight.data._local_tensor

            # Case 1: Replicated weights
            if sharded_layout == 'REPLICATED':
                full_weight = local_weight
            # Case 2: Colwise sharded weights
            elif sharded_layout == 'COLWISE':
                # Gather everything
                if dp_rank == 0:
                    full_weight = gather_weights(tp_rank, local_weight, sharded_dims['tp_dim'], device_mesh.get_group(mesh_dim="tp"))

            # Case 3: Rowwise sharded weights
            elif sharded_layout == 'ROWWISE':
                if dp_rank == 0:
                    full_weight = gather_weights(tp_rank, local_weight, sharded_dims['tp_dim'], device_mesh.get_group(mesh_dim="tp"))

            else:
                raise AssertionError('Unknown sharded layout detected')

            if global_rank == 0:
                cpu_state_dict[peft_weight].copy_(full_weight)

                # rank_print(rank, f'{rank=} {peft_weight=} {full_weight=}')
        # rank_print(rank, f'{rank=} {peft_weight=} after: {cpu_state_dict[peft_weight]=}')

    rank_print(rank, 'Adapters moved to cpu')
    # # sleep 30s
    # if global_rank !=0:
    #     time.sleep(30)
    dist.barrier()

def move_adapters_to_cpu_for_FSDP(rank,
                                  model,
                                  cpu_state_dict,
                                  device_mesh
                                  ):
    '''
    IMPORTANT FUNCTION: used by official training runs
    '''
    rank_print(rank, 'Moving adapters to cpu')
    peft_weights = [weight_name for weight_name in cpu_state_dict if 'lora' in weight_name]
    global_rank = rank

    for peft_weight in tqdm(peft_weights):
        decoder_block_path, rest_of_path = get_decoder_block_path(peft_weight)

        if decoder_block_path is not None:
            module = get_nested_attribute(model, decoder_block_path)
        else:
            module = model
            rest_of_path = peft_weight

        # rank_print(rank, f'{rank=} {peft_weight=} before: {cpu_state_dict[peft_weight]=}')
        with _unshard_params(module,
                            recurse=False,
                            writeback=False,
                            rank0_only=False,
                            offload_to_cpu=False,
                            with_grads=False,):
            weight = get_nested_attribute(module, rest_of_path)
            full_weight = weight.data

            if global_rank == 0:
                cpu_state_dict[peft_weight].copy_(full_weight)

    rank_print(rank, 'Adapters moved to cpu')
    # # sleep 30s
    # if global_rank !=0:
    #     time.sleep(30)
    dist.barrier()

def move_adapters_to_cpu_for_FSDP2(
    rank,
    model,
    cpu_state_dict,
    adapter_name_to_export: str = 'default',
    **kwargs,
):
    '''
    IMPORTANT FUNCTION: used by official training runs
    Move only lora weights from GPU-sharded model into CPU state dict and save them.

    Args:
        rank: The rank of the current process.
        model: The GPU model to move adapters from.
        cpu_state_dict: The cpu state dict to load the adapters into.
        kwargs: Additional arguments.

    Returns:
        None
    '''
    rank_print(rank, 'Moving adapters to cpu')
    cpu_peft_weight_names = [weight_name for weight_name in cpu_state_dict if 'lora' in weight_name and '.default' in weight_name]

    for cpu_peft_weight_name in tqdm(cpu_peft_weight_names):
        model_peft_weight_name = cpu_peft_weight_name.replace('default', adapter_name_to_export)
        try:
            full_weight = get_nested_attribute(model, model_peft_weight_name).full_tensor().detach()
        except AttributeError:
            full_weight = get_nested_attribute(model, model_peft_weight_name).detach()

        if rank == 0:
            cpu_state_dict[cpu_peft_weight_name].copy_(full_weight)

    rank_print(rank, 'Adapters moved to cpu')
    dist.barrier()

def load_adapters_from_cpu_state_dict_for_FSDP(model, cpu_state_dict, rank, device_mesh):
    print("Loading adapters from cpu_state_dict ...")
    for module_name in tqdm(cpu_state_dict):
        if 'lora' in module_name:
            load_from_state_dict_for_FSDP_TP(model,
                                module_name,
                                rank,
                                device_mesh,
                                cpu_state_dict[module_name])
    print(f'Adapter weights loaded on {rank=}')

def load_adapters_from_cpu_state_dict(model, cpu_state_dict, rank, tp_mesh):
    print("Loading adapters from cpu_state_dict ...")

    module_list = [str(module_name) for module_name in cpu_state_dict]
    # sort the module list to make sure the order is the same across all ranks
    module_list = sorted(module_list)

    for module_name in tqdm(module_list):
        if 'lora' in module_name:
            load_from_state_dict(model,
                                module_name,
                                rank,
                                tp_mesh,
                                cpu_state_dict[module_name])

    print(f'Adapter weights loaded on {rank=}')

def save_gradients_fsdp(
    rank,
    model,
    save_dir: Union[str, Path],
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for module_name in tqdm(model.state_dict()):
        decoder_block_path, rest_of_path = get_decoder_block_path(module_name)
        if decoder_block_path is not None:
            module = get_nested_attribute(model, decoder_block_path)
        else:
            module = model
            rest_of_path = module_name

        with _unshard_params(module,
            recurse=False,
            writeback=False,
            rank0_only=True,
            offload_to_cpu=False,
            with_grads=True,
        ):
            if get_nested_attribute(module, rest_of_path).requires_grad:
                if rank == 0:
                    grad = get_nested_attribute(module, rest_of_path).grad.detach()
                    torch.save(grad, Path() / save_dir / f'{module_name}.pt')
            else:
                continue

    dist.barrier()
    print(f'All gradients saved at {save_dir}')
    return

def save_gradients_fsdp2(
    rank,
    model,
    save_dir: str,
):
    print('Saving gradients ...')
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for module_name in tqdm(model.state_dict()):
        if get_nested_attribute(model, module_name).requires_grad:
            full_grad = get_nested_attribute(model, module_name).grad.full_tensor().detach()
            if rank == 0:
                torch.save(full_grad, save_dir / f'{module_name}.pt')
        else:
            continue

    dist.barrier()
    print(f'All gradients saved at {save_dir}')
    return

def save_weights_fsdp(
    rank,
    model,
    save_dir: Union[str, Path],
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for module_name in tqdm(model.state_dict()):
        decoder_block_path, rest_of_path = get_decoder_block_path(module_name)
        if decoder_block_path is not None:
            module = get_nested_attribute(model, decoder_block_path)
        else:
            module = model
            rest_of_path = module_name

        with _unshard_params(module,
            recurse=False,
            writeback=False,
            rank0_only=True,
            offload_to_cpu=False,
            with_grads=False,
        ):
            if get_nested_attribute(module, rest_of_path).requires_grad:
                if rank == 0:
                    grad = get_nested_attribute(module, rest_of_path).detach()
                    torch.save(grad, Path() / save_dir / f'{module_name}.pt')
            else:
                continue

    dist.barrier()
    print(f'All gradients saved at {save_dir}')
    return

def save_weights_fsdp2(
    rank,
    model,
    save_dir: str,
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for module_name in tqdm(model.state_dict()):
        if get_nested_attribute(model, module_name).requires_grad:
            full_weight = get_nested_attribute(model, module_name).full_tensor().detach()
            if rank == 0:
                torch.save(full_weight, save_dir / f'{module_name}.pt')
        else:
            continue

    dist.barrier()
    print(f'All weights saved at {save_dir}')
    return
