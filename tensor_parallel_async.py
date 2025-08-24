from functools import partial
from typing import Any , Callable , Dict , List,  Literal , Sequence , Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict

from pipeline_parallel import ModelParallelWrapper
from tensor_paral import MLPBlockInput , MLPBlockOutput , scale_init


PyTree = Any
Parameter = jax.Array | nn.Partitioned
Metrics = Dict[str, Tuple[jax.Array, ...]]


def async_gather(
    x  : PyTree,
    axis_name : str,
    shift_up : bool = True
) -> List[PyTree]:

    tp_size = jax.lax.psum(1, axis_name)
    if shift_up:
        shift_perm = [(j, (j + 1) % tp_size) for j in range(tp_size)]
    else:
        shift_perm = [(j, (j - 1) % tp_size) for j in range(tp_size)]
    ps = [x]
    p = x
    for _ in range(1, tp_size):
            p = jax.lax.ppermute(p, axis_name, perm=shift_perm)
            ps.append(p)
        return ps