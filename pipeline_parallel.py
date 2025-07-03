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
from jax.tree_util import tree_map
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


config = ConfigDict(
    dict(
        dtype = jnp.bfloat16
    )
)


class MLPBlock(nn.Module):

    config : ConfigDict
    train : bool

    @nn.compact
    def __call__(
        self,
        x : jax.Array
    ) -> jax.Array:

        input_feat = x.shape[-1]
        residual = x
        x = nn.LayerNorm(
            dtype=self.config.dtype,
            name="pre_norm"
        )(x)

        x = nn.Dense(
            features= self.config.hidden_size * self.config.mlp_expansion,
            dtype=self.config.dtype,
            name="input_dense"
        )(x)

        x = nn.silu(x)
        x = nn.Dropout(rate=self.config.dropout_rate , deterministic=not self.train)(x)
        x = nn.Dense(features=input_feat , dtype=self.config.dtype , name="output")(x)

        return x + residual


class MLPLayers(nn.Module):
    config: ConfigDict
    train: bool

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        # Scan version
        block_class = MLPBlock
        if "MLP" in self.config.remat:
            block_class = nn.remat(block_class, prevent_cse=False)
        block = block_class(config=self.config, train=self.train, name="block")
        x, _ = nn.scan(
            lambda module, carry, _: (module(carry), ()),
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            length=self.config.num_layers,
        )(block, x, ())
        
        return x


def stack_params(
        params : PyTree,
        axis_name : str,
        axis : int = 0,
        mask_except : jax.Array | int | None
    ) -> PyTree:


    def _stack(x):

        if isinstance(x , nn.Partitioned):

            value , names = x.value , x.names

        else:
            value , names = x , (None ,) * x.ndim

        if mask_except is not None:

            axis_index = jax.lax.axis_index(axis_name)
            value = jnp.expand_dims(value , axis)

        value = jnp.expand_dims(value , axis)
        names = names[:axis] + (axis_name , ) + names[axis + 1 :]

        return nn.Partitioned(value , names=names)
    
    return tree_map(
        _stack,
        params,
        is_leaf= lambda x : isinstance(x , nn.Partitioned)
    )



def unstack_params(
        params : PyTree,
        axis_name : str
):
    def _unstack(x: Parameter) -> Parameter:
        if isinstance(x, nn.Partitioned) and axis_name in x.names:
            value = x.value
            names = x.names
            axis_idx = names.index(axis_name)
            value = value.squeeze(axis_idx)
            names = names[:axis_idx] + names[axis_idx + 1 :]
            if all([n is None for n in names]):
                return value
            else:
                return nn.Partitioned(value, names=names)
        else:
            return x
        
    
    return tree_map(
        _unstack,
        params,
        is_leaf= lambda x : isinstance(x , nn.Partitioned)

    )
        


        
