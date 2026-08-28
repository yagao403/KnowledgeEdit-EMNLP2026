from dataclasses import dataclass
from typing import ClassVar
import torch

@dataclass
class TorchConfigMini:
    model_id: str = "meta-llama/Meta-Llama-3.1-70B-Instruct"  # TODO
    # model_id: str="meta-llama/Llama-3.2-1B-Instruct"
    layer_name: str = 'base_model.model.model.layers.0.self_attn.v_proj.lora_B.default' # Layer whoses weights are to be saved
    # layer_name: str = 'lm_head'
    mixed_precision: bool = True
    use_fp16: bool = False
    num_hidden_layers: int = None
    torch_dtype: str = 'bf16' # 'bf16' or 'f32'
    gradient_checkpointing: bool = True
    first_run: bool = True
    save_and_load_optimizer_state: bool = False
    save_adapters: bool = False
    use_peft: bool = True
    sharded_load: bool = True
    reset_adapters: bool = True
    adapter_name: str = 'default'  # TODO
    adapter_dir: str = 'adapters'  # TODO
