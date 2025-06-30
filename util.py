import jax
from flax.struct import dataclass
from flax.training import train_state
from typing import Any , Callable , Dict , Tuple

import os

Metrics = Dict[str , Tuple[jax.Array , ...]]

class TrainState(train_state.TrainState):
    rng : jax.Array


@dataclass
class Batch:
    inputs : jax.Array
    labels : jax.Array


def sim_multiCPU_dev(
        device_count : int = 8
):    
    flags = os.environ.get("XLA_FLAGS" , "")
    flags += f" --xla_force_host_platform_device_count={device_count}"
    
    os.environ["XLA_FLAGS"] = flags
    
    #disable CUDA 
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

