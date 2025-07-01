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

from util import Batch , TrainState , accum_grads , Pytree , Metrics


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
    
# init
model_dp = Classifier(config=CONFIG.model)
optimizer = optax.adamw(
    learning_rate=CONFIG.OPTIMIZER_CONFIG.learning_rate,
)

rng =  jax.random.PRNGKey(CONFIG.seed)

model_init_rng , data_inp_rng , datal_label_rng = jax.random.split(rng , 3)
batch = Batch(
    inputs= jax.random.normal(
        data_inp_rng,
        (CONFIG.DATA_CONFIG.batch_size , CONFIG.DATA_CONFIG.input_size) # type: ignore
    ),
    labels = jax.random.normal(
        datal_label_rng , 
        (CONFIG.DATA_CONFIG.batch_size,),
        0,
        CONFIG.DATA_CONFIG.num_classes
    ),
)



def init_dp(
        rng,
        x : jax.Array,
        model : nn.Module
) -> TrainState:
    """
    function to initialize training setup
    """
    init_rng , rng = jax.random.split(rng)
    var = model.init({"params" : init_rng} , x , train=False)
    params = var.pop("params")

    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
        rng=rng,
    )

    return state


devices = np.array(jax.devices())
#init mesh
mesh = Mesh(devices , (CONFIG.data_axis_name,))

#initialize sharding
init_dp_fn = jax.jit(
    shard_map(
        functools.partial(init_dp , model=model_dp),
        mesh,
        in_specs=(P() , P(CONFIG.data_axis_name)),
        out_specs=P(),
        check_rep=False,
    ),
)

state_dp = init_dp_fn(
    model_init_rng,
    batch.inputs
)


def loss_fn(
        params,
        apply_fn,
        batch,
        rng
):
    
    dropout_rng = fold_rng_over_axis(rng , CONFIG.data_axis_name)
    logits = apply_fn({"params" : params} , batch.inputs , train=True , rngs={"dropout" : dropout_rng})

    loss = optax.softmax_cross_entropy_with_integer_labels(logits , batch.labels)

    correct_pred = jnp.equal(jnp.argmax(logits , axis=-1) , batch.labels)

    bs =batch.inputs.shape[0]
    step_metrics = {"loss" : (loss.sum() , bs) , "accuracy" : (correct_pred.sum() , bs)}
    loss = loss.mean()

    return loss , step_metrics






        
