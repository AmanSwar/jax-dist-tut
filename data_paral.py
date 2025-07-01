from util import sim_multiCPU_dev
#simulate multiple devices on CPU
sim_multiCPU_dev()


import functools
from pprint import pprint
from typing import Any , Callable , Dict,  Sequence, Tuple


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

from util import Batch , TrainState , accum_grads 


def fold_rng_over_axis(
        rng : jax.random.PRNGKey,
        axis_name : str
):
    
    axis_index = jax.lax.axis_index(axis_name)
    return jax.random.fold_in(rng , axis_index)






class Classifier(nn.Module):

    config : ConfigDict

    @nn.compact
    def __call__(
        self,
        x : jax.Array,
        train : bool
    ) -> jax.Array:
        
        x = nn.Dense(
            features=self.config.hidden_size,
            dtype=self.config.dtype,
            name="input_dense",
        )(x)

        x = nn.silu(x)
        x = nn.Dropout(rate= self.config.dropout_rate , deterministic=not train)(x)
        x = nn.Dense(
            features=self.config.num_classes,
            dtype=self.config.dtype,
            name="output_dense"
        )(x)

        x = x.astype(jnp.float32)

        return x
    


        
