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



#configs
DATA_CONFIG = ConfigDict(
    dict(
        batch_size=128,
        num_classes=10,
        input_size=784,
    )
)

MODEL_CONFIG = ConfigDict(
    dict(
        hidden_size=512,
        dropout_rate=0.1,
        dtype=jnp.bfloat16,
        num_classes=DATA_CONFIG.num_classes,
        data_axis_name="data",
    )
)

OPTIMIZER_CONFIG = ConfigDict(
    dict(
        learning_rate=1e-3,
        num_minibatches=4,
    )
)

CONFIG = ConfigDict(
    dict(
        model= MODEL_CONFIG,
        optimizer = OPTIMIZER_CONFIG,
        data = DATA_CONFIG,
        data_axis_name=MODEL_CONFIG.data_axis_name,
        seed=69,
    )
)


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
    


        
