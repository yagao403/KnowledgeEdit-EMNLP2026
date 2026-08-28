from dataclasses import dataclass
from typing import Literal

@dataclass
class LoRAConfig:
    lora_target_modules: Literal['full', 'qk', 'qv'] = 'full'
    lora_r: int = 128 # 32, 64, 128, 256, 512, 1024
