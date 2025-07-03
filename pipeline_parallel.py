import jax.experimental
import jax.experimental.shard_map
from util import sim_multiCPU_dev

# simulate multiple devices on CPU
sim_multiCPU_dev()

import functools
from pprint import pprint
from typing import Any , Callable , Dict , Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core.frozen_dict import FrozenDict
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from ml_collections import ConfigDict


PyTree = Any
Parameter = jax.Array | nn.Partitioned
Metrics = Dict[str , Tuple[jax.Array , ...]]


from single_gpu import (
    Batch,
    TrainState,
    accumulate_gradients,
    get_num_params,
    print_metrics
)

from data_paral import fold_rng_over_axis
from param_sharding import sync_gradients

