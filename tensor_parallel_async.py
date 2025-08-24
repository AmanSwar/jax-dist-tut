from functools import partial
from typing import Any , Callable , Dict , List,  Literal , Sequence , Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict

from pipeline_parallel import ModelParallelWrapper
from tensor_paral import MLPBlockInput , MLPBlockOutput , scale_init


