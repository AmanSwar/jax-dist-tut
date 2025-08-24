from functools import partial
from typing import Any, Callable, Dict, Tuple


import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict

from data_paral import shard_module_params
from pipeline_parallel import ModelParallelWrapper
from tensor_paral import MLPBlockInput , MLPBlockOutput
from tensor_parallel_async import TPAsyncDense ,TPAsyncMLPBlock , TPNorm



