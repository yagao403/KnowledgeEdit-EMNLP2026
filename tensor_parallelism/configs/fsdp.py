from dataclasses import dataclass, field
from typing import ClassVar
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

@dataclass
class fsdp_config:
    limit_all_gathers: bool=True
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD # FULL_SHARD, HYBRID_SHARD, SHARD_GRAD_OP
    checkpoint_type: StateDictType = StateDictType.SHARDED_STATE_DICT # alternatively can use SHARDED_STATE_DICT to avoid OOMs # FULL_STATE_DICT, SHARDED_STATE_DICT
