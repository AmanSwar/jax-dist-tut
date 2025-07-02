import functools
from pprint import pprint
from typing import Any, Callable, Dict, Sequence, Tuple


import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from jax import lax
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from ml_collections import ConfigDict
from absl import logging

from util import Pytree


@jax.named_scope("shard_params")
def shard_params(
    params: Pytree, axis_name: str, min_weight_size: int = 2**18
) -> Pytree:

    axis_idx = jax.lax.axis_index(axis_name)
    # find axis size using psum
    axis_size = jax.lax.psum(1, axis_name)

    # function to apply partion
    def _split(x):
        # cond1 : if already Partitioned
        if isinstance(x, nn.Partitioned):

            values, names = x.value, x.names

        else:
            value = x
            names = (None,) * value.ndim

        # condition to check if value is already shared or not
        if axis_name in names:
            logging.warning(
                f"Param {value.shape} with name {names} alread shareded on axis {axis_name}"
            )
            return x

        # case if value is less then minimum required value to be shareded
        elif value.size <= min_weight_size:
            logging.info(f"Param {value.shape} too small")

            return x

        else:
            shape = value.shape
            idx = np.argsort(shape)[::-1]  # shape in descending order of largest axis

            for i in idx:

                if shape[i] % axis_size == 0 and names[i] is None:

                    split_size = shape[i] // axis_size
                    p_sharded = nn.Partitioned(
                        value=lax.dynamic_slice_in_dim(
                            value, axis_idx * split_size, split_size, axis=i
                        ),
                        names=names[:i] + (axis_name,) + names[i + 1 :],
                    )

                    return p_sharded

            logging.warning(
                f"could not shard {value.shape} on axis {axis_name} cuz no suitable axis found"
            )
            return x

    # apply _split to each leaf node of PyTree
    return jax.tree_util.tree_map(
        _split,
        params,
        is_leaf=lambda x: isinstance(x, nn.Partitioned),
    )
