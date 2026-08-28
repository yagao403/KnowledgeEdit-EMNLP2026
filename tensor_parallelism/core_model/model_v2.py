from transformers import AutoModelForCausalLM
from transformers.configuration_utils import PretrainedConfig
import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy, OffloadPolicy
from peft import get_peft_model, PeftModel, PeftMixedModel, PeftConfig
from functools import partial
from pathlib import Path
import time
from typing import Union
from termcolor import colored
import warnings

from tensor_parallelism.utils import (
    get_nested_attribute,
    str_to_dtype,
    load_weight_from_state_dict_and_buffer_for_FSDP2,
)

def parallelize_peft_model_fsdp2(
    model: Union[PeftModel, PeftMixedModel],
    device_mesh: DeviceMesh,
    torch_dtype: torch.dtype,
):
    '''
    IMPORTANT FUNCTION: used by both experimental runs and official training runs
    '''
    # Take a sample lora layer
    for name, module in model.named_modules():
        if "lora_A" in name:
            lora_layer = module
            break

    if lora_layer.default.weight.dtype != torch_dtype:  # type: ignore
        warning_msg = f"[WARNING] LoRA weight dtype {lora_layer.weight.dtype} does NOT match expected dtype {torch_dtype}"
        print(colored(warning_msg, 'red', attrs=['bold']))
        warnings.warn(warning_msg, stacklevel=2)

    # Apply FSDP2 to the model
    # Since fully_shard() will wrap the input module as a single FSDPModule and if we call .unshard()
    # it will all-gather all the managed params. Therefore, we have to apply fully_shard() in a bottom-up manner.
    # By doing that, we wrap each submodule, in this case, the decoder block, as a separate FSDPModule.
    # When we apply fully_shard() to the root module, i.e. the model,
    # it only wraps non-FSDPModule params together into a single FSDPModule, traversing recursively.

    # Define Mixed Precision policy
    mixed_precision_policy = MixedPrecisionPolicy()

    # Apply FSDP2 to each decoder block
    layers = list(model.base_model.model.model.layers)  # type: ignore

    for _, llama_decoder_block in enumerate(layers):
        fully_shard(
            llama_decoder_block,
            mesh=device_mesh,
            reshard_after_forward=True,
            mp_policy=mixed_precision_policy,
            offload_policy=OffloadPolicy(), # OffloadPolicy() means no offloading
        )

    # Finally, apply FSDP2 to non-decoder components (i.e., other layers that are not decoder blocks, e.g., embed_tokens, norm, lm_head)
    fully_shard(  # type: ignore
        model.base_model.model.model,  # type: ignore
        mesh=device_mesh,
        reshard_after_forward=True,
        mp_policy=mixed_precision_policy,
        offload_policy=OffloadPolicy(), # OffloadPolicy() means no offloading
    )

    fully_shard(  # type: ignore
        model.base_model.model.lm_head,  # type: ignore
        mesh=device_mesh,
        reshard_after_forward=True,
        mp_policy=mixed_precision_policy,
        offload_policy=OffloadPolicy(), # OffloadPolicy() means no offloading
    )

    # Safety Protocol: Applying fully_shard() on the root module although at this point there is no param left.
    fully_shard(
        model,
        mesh=device_mesh,
        reshard_after_forward=True,
        mp_policy=mixed_precision_policy,
        offload_policy=OffloadPolicy(), # OffloadPolicy() means no offloading
    )

    return model

def parallelize_model_fsdp2(
    model: AutoModelForCausalLM,
    device_mesh: DeviceMesh,
    torch_dtype: torch.dtype,
):
    # Apply FSDP2 to the model
    # Since fully_shard() will wrap the input module as a single FSDPModule and if we call .unshard()
    # it will all-gather all the managed params. Therefore, we have to apply fully_shard() in a bottom-up manner.
    # By doing that, we wrap each submodule, in this case, the decoder block, as a separate FSDPModule.
    # When we apply fully_shard() to the root module, i.e. the model,
    # it only wraps non-FSDPModule params together into a single FSDPModule, traversing recursively.

    # Define Mixed Precision policy
    mixed_precision_policy = MixedPrecisionPolicy()
    print(f'{mixed_precision_policy=}')
    # Apply FSDP2 to each decoder block
    layers = list(model.model.layers)  # type: ignore

    # Apply FSDP2 to each decoder block
    for _, llama_decoder_block in enumerate(layers):
        fully_shard(
            llama_decoder_block,
            mesh=device_mesh,
            reshard_after_forward=True,
            mp_policy=mixed_precision_policy,
            offload_policy=OffloadPolicy(), # OffloadPolicy() means no offloading
        )

    # Finally, apply FSDP2 to non-decoder components (i.e., other layers that are not decoder blocks, e.g., embed_tokens, norm, lm_head)
    fully_shard(
        model.model,
        mesh=device_mesh,
        reshard_after_forward=True,
        mp_policy=mixed_precision_policy,
        offload_policy=OffloadPolicy(), # OffloadPolicy() means no offloading
    )

    # For original model, we don't need to wrap lm_head separately as in PEFT. Not sure why.
    # We want to minimize the amount of unnecessary sharding code.
    fully_shard(
        model.lm_head,
        mesh=device_mesh,
        reshard_after_forward=True,
        mp_policy=mixed_precision_policy,
        offload_policy=OffloadPolicy(), # OffloadPolicy() means no offloading
    )

    # Safety Protocol: call fully_shard() on the model although there's no non-fsdp params left
    fully_shard(
        model,
        mesh=device_mesh,
        reshard_after_forward=True, # Set to False as this only contains lm_head, which wil be used right after forward to compute gradients
        mp_policy=mixed_precision_policy,
        offload_policy=OffloadPolicy(), # OffloadPolicy() means no offloading
    )
    return model

def load_cpu_model(
    model_id: str | Path,
    model_config: PretrainedConfig,
    adapter_id: str | Path | None = None,
    adapter_name: str = "default",
    torch_dtype: torch.dtype = torch.bfloat16,
    peft_config: PeftConfig | None = None,
    quantize: bool = False,
    merge: bool = True,
) -> AutoModelForCausalLM | PeftModel | PeftMixedModel:
    # Load model into cpu and extract state_dict and buffers
    t0 = time.perf_counter()

    cpu_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=model_config,
        device_map='cpu',
        load_in_8bit=quantize,
        torch_dtype=torch_dtype,
        #attn_implementation="flash_attention_2"
        trust_remote_code=True,
    )
    t = time.perf_counter() - t0
    print(f"Time to load the model: {t:.02f} sec", flush=True)

    # Merge adapters into the model or create PEFT model
    if adapter_id:
        t0 = time.perf_counter()
        cpu_model = PeftModel.from_pretrained(cpu_model, adapter_id, adapter_name, is_trainable=True, torch_device='cpu')
        if merge:
            print(f'Merging adapter {adapter_id} into the model')
            cpu_model = cpu_model.merge_and_unload()  # type: ignore
            t = time.perf_counter() - t0
        else:
            print(f"No merging of adapter {adapter_id}")

    if peft_config is not None:
        if adapter_id is not None and not merge:
            print("adapter_id is already given, loaded and not merged. "
                  "Using the loaded adapters.")
        else:
            cpu_model = get_peft_model(
                model=cpu_model,
                peft_config=peft_config,
                adapter_name=adapter_name,
                autocast_adapter_dtype=False
            )

    return cpu_model

def load_base_model(
    model_id: str | Path,
    model_config: PretrainedConfig,
    torch_dtype: torch.dtype = torch.bfloat16,
    training: bool = False,
    device_map: str | None = None,
    gradient_checkpointing: bool = False,
    transformer_layer_cls: type[torch.nn.Module] | tuple[type[torch.nn.Module], ...] | None = None,
):
    if training and device_map == "auto":
        raise ValueError(
            "device_map='auto' is not supported for training."
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=model_config,
        device_map=device_map,
        load_in_8bit=False,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )

    print("Model loaded.")

    if gradient_checkpointing:
        print('Using built-in gradient checkpointing enabler.')
        model.gradient_checkpointing_enable({'use_reentrant': False})
        # Legacy code: Comment out the line above and uncomment the following code
        # to manually apply AC on transformer layers
        # apply_activation_checkpointing(
        #     model,
        #     checkpoint_wrapper_fn=partial(
        #         checkpoint_wrapper,
        #         checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        #     ),
        #     check_fn=lambda m: isinstance(m, transformer_layer_cls),
        # )
    return model

def create_FSDP2_parallelized_model(
    rank: int,
    *,
    device_mesh: DeviceMesh,
    model_id: str | Path,
    model_config: PretrainedConfig,
    adapter_id: str | Path | None = None,
    adapter_name: str = "default",
    torch_dtype: torch.dtype = torch.bfloat16,
    peft_config: PeftConfig | None = None,
    gradient_checkpointing: bool = False,
    transformer_layer_cls: type[torch.nn.Module] | tuple[type[torch.nn.Module], ...] | None = None,
    device_map: str | None = None,
    training: bool = True,
    efficient_loading: bool = True,
    quantize: bool = False,
    merge: bool = True,
    ema: bool = False,  # Whether to create an extra adapter to save EMA weights
):
    '''
    Create FSDP2 parallelized model that load the whole model onto each GPU.

    Parameters:
        rank: rank of the sharded model. This must always be the local rank even when running on multiple nodes.
        device_mesh: device mesh
        model_id: identifier of the model to load (could be a path or a model name)
        adapter_id: directory of trained adapters
        adapter_name: name of the adapter to load
        torch_dtype: dtype of the model
        peft_config: PEFT configuration
        gradient_checkpointing: whether to enable gradient checkpointing or not
        transformer_layer_cls: transformer layer class to apply gradient checkpointing on
        device_map: device map to use for the model. If None, the model will be loaded onto the current device.
        training: whether to create a trainable model.
        quantize: whether to quantize the model or not
        merge: whether to merge the adapter into the model or not

    Returns:
        model: FSDP2 parallelized model
    '''

    tp_mesh = device_mesh
    local_rank = tp_mesh.get_local_rank()
    print(f'{local_rank=}')

    if efficient_loading:
        cpu_model = load_cpu_model(
            model_id=model_id,
            model_config=model_config,
            adapter_id=adapter_id,
            adapter_name=adapter_name,
            torch_dtype=torch_dtype,
            peft_config=peft_config,
            quantize=quantize,
            merge=merge,
        )
        if peft_config is None and isinstance(cpu_model, (PeftModel, PeftMixedModel)) \
            and cpu_model.peft_config is not None:
            peft_config = cpu_model.peft_config['default']

        if ema:
            assert peft_config is not None
            assert isinstance(cpu_model, (PeftModel, PeftMixedModel))
            cpu_model.add_adapter(adapter_name="ema", peft_config=peft_config)
            for p_name, param in cpu_model.named_parameters():
                if "lora" in p_name and ".ema." in p_name:
                    param.data = param.data.to(device='cpu', dtype=torch_dtype)
            # Initialize the EMA adapter with the same weights as the default model
            cpu_state_dict = cpu_model.state_dict()
            for p_name, param in cpu_model.named_parameters():
                if "lora" in p_name and "default" in p_name:
                    ema_name = p_name.replace("default", "ema")
                    ema_param = cpu_state_dict[ema_name]
                    ema_param.data.copy_(param.data)

        cpu_state_dict = cpu_model.state_dict()  # type: ignore

        cpu_buffers = {}
        for buffer_name, module_buffer in cpu_model.named_buffers():  # type: ignore
            cpu_buffers[buffer_name] = module_buffer

    if efficient_loading and device_map != "meta":
        print(
            "WARNING: efficient_loading is enabled but device_map is not set to 'meta'. "
            "Automatically set device_map='meta'."
        )
        device_map = "meta"

    # Create FSDP2 parallelized model
    model = load_base_model(
        model_id=model_id,
        model_config=model_config,
        torch_dtype=str_to_dtype(torch_dtype),
        training=training,
        device_map=device_map,
        gradient_checkpointing=gradient_checkpointing,
        transformer_layer_cls=transformer_layer_cls if gradient_checkpointing else None,
    )

    if peft_config is not None:
        model = get_peft_model(
            model,
            peft_config,
            autocast_adapter_dtype=False
        )
        if ema:
            assert isinstance(model, (PeftModel, PeftMixedModel))
            model.add_adapter(adapter_name="ema", peft_config=peft_config, low_cpu_mem_usage=True)
            for p_name, param in model.named_parameters():
                if "lora" in p_name and ".ema." in p_name:
                    param.data = param.data.to(device=param.data.device, dtype=torch_dtype)
                    param.requires_grad = False

        if device_map == "auto":
            raise ValueError(
                "Cannot apply FSDP2 to a model with device_map='auto'. "
            )

        model = parallelize_peft_model_fsdp2(
            model=model,
            device_mesh=device_mesh,
            torch_dtype=torch_dtype
        )

        # IMPORTANT: Freezing the base model weights as PEFT does not handle it automatically when the model is set to 'meta' device
        peft_freeze_weights = [weight_name for weight_name in model.state_dict() if 'lora' not in weight_name]
        for weight_name in peft_freeze_weights:
            get_nested_attribute(model, weight_name).requires_grad = False # type: ignore

    else:
        model = parallelize_model_fsdp2(
            model=model,
            device_mesh=device_mesh,
            torch_dtype=torch_dtype
        )

    if efficient_loading:
        # Move the model to the current device
        model.to_empty(device=torch.cuda.current_device())

        load_weight_from_state_dict_and_buffer_for_FSDP2(
            rank,
            device_mesh=device_mesh,
            model=model,
            cpu_state_dict=cpu_state_dict,
            cpu_buffers=cpu_buffers,
        )

    if hasattr(model, 'print_trainable_parameters'):
        model.print_trainable_parameters()  # type: ignore

    return model
