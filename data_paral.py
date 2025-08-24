from util import sim_multiCPU_dev
# simulate multiple devices on CPU
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

from util import Batch , TrainState , accum_grads , Pytree , Metrics , print_metrics

PyTree = Any
Parameter = jax.Array | nn.Partitioned
Metrics = Dict[str , Tuple[jax.Array , ...]]

def fold_rng_over_axis(
        rng : jax.random.PRNGKey,
        axis_name : str
):
    
    axis_index = jax.lax.axis_index(axis_name)
    return jax.random.fold_in(rng , axis_index)


# configs
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
    Function to initialize model on each device

    Args:
        rng (jax.random.PRNGkey): key 
        x (jax.Array): input
        model (nn.Module): model to be initialized
    Returns:
        TrainState : state
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
# init mesh
mesh = Mesh(devices , (CONFIG.data_axis_name,))

# initialize sharding
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


def train_step_dp(
        state: TrainState,
        metrics : Metrics,
        batch : Batch
)-> Tuple[TrainState , Metrics]:
    
    rng , step_rng = jax.random.split(state.rng)

    grads , step_metrics = accum_grads(
        state,
        batch,
        step_rng,
        CONFIG.optimizer.num_minibatches,
        loss_fn=loss_fn
    )


    with jax.named_scope("sync_grads"):
        grads = jax.tree_util.tree_map(
            lambda g : jax.lax.pmean(g , axis_name=CONFIG.data_axis_name) , grads)
        
    new_state = state.apply_gradients(
        grads=grads,
        rng=rng
    )


    with jax.named_scope("sync_metrics"):

        step_metrics = jax.tree_util.tree_map(
            lambda x : jax.lax.psum(
                x , 
                axis_name=CONFIG.data_axis_name
            ),
                step_metrics
        )

    if metrics is None:
        metics = step_metrics

    else:
        metrics = jax.tree_util.tree_map(
            jnp.add , metrics , step_metrics
        )

    return new_state , metrics


train_step_dp_fn = jax.jit(
    shard_map(
        train_step_dp,
        mesh,
        in_specs=(P() , P() , P(CONFIG.data_axis_name)),
        out_specs=(P() , P()),
        check_rep=False,
    ),

    donate_argnames=("state" , "metrics"),
)


@jax.named_scope("shard_params")
def shard_params(
    params: PyTree, axis_name: str, min_weight_size: int = 2**18
) -> PyTree:

    axis_idx = jax.lax.axis_index(axis_name)
    axis_size = jax.lax.psum(1, axis_name)

    def _split(x: Parameter) -> Parameter:
        if isinstance(x, nn.Partitioned):
            value, names = x.value, x.names
        else:
            value = x
            names = (None,) * value.ndim
        if axis_name in names:
            logging.warning(
                f"Parameter {value.shape} with names {names} already sharded on axis {axis_name}."
            )
            return x
        elif value.size <= min_weight_size:
            logging.info(
                f"Parameter {value.shape} with names {names} too small to shard, size {value.size} < {min_weight_size}."
            )
            return x
        else:
            shape = value.shape
            idx = np.argsort(shape)[::-1]  # Shard along largest possible axis.
            for i in idx:
                if shape[i] % axis_size == 0 and names[i] is None:
                    split_size = shape[i] // axis_size
                    p_sharded = nn.Partitioned(
                        value=lax.dynamic_slice_in_dim(  # Shard to keep on present device.
                            value, axis_idx * split_size, split_size, axis=i
                        ),
                        names=names[:i] + (axis_name,) + names[i + 1 :],
                    )
                    return p_sharded
            logging.warning(
                f"Could not shard {value.shape} with names {names} on axis {axis_name}, no suitable axis found."
            )
            return x

    return jax.tree_util.tree_map(
        _split,
        params,
        is_leaf=lambda x: isinstance(
            x, nn.Partitioned
        ),  # Consider a nn.Partitioned object as a leaf.
    )


def gather_array_with_mean_grads(x: jax.Array, axis: int, axis_name: str):
    axis_size = jax.lax.psum(1, axis_name)

    # Define a custom gradient for the gather operation.
    @jax.custom_gradient
    def f(x):
        def grad_fn(g):
            # pmean_scatter
            return (
                jax.lax.psum_scatter(g, axis_name, scatter_dimension=axis, tiled=True)
                / axis_size
            )

        return jax.lax.all_gather(x, axis_name, axis=axis, tiled=True), grad_fn

    return f(x)




def sync_grads(
    grads : Pytree,
    axis_names = Sequence[str]
):
    def _sync_grads(g : Parameter):

        if isinstance(g , nn.Partitioned):
            replication_axis_name = [
                name for name in axis_names if name not in jax.tree_util.tree_leaves(g.names)
            ]


            if len(replication_axis_name) == 0:
                return g
            

            else:
                return g.replace(
                    value=jax.lax.pmean(g.value , axis_name=replication_axis_name)
                )
        else:
            return jax.lax.pmean(g , axis_name=axis_names)
        

    return jax.tree_util.tree_map(
        _sync_grads , grads , is_leaf= lambda x : isinstance(x , nn.Partitioned)
    )


_ , metric_shape = jax.eval_shape(
    train_step_dp_fn,
    state_dp,
    None,
    batch
)


metrics_dp = jax.tree_util.tree_map(
    lambda x : jnp.zeros(
        x.shape,
        dtype=x.dtype
    ),
    metric_shape,)    

# main loop
for _ in range(10):

    state_dp , metrics_dp = train_step_dp_fn(state_dp , metrics_dp , batch)

final_metrics_dp = jax.tree_util.tree_map(lambda x: jnp.zeros(x.shape, dtype=x.dtype), metric_shape)
state_dp, final_metrics_dp = train_step_dp_fn(state_dp, final_metrics_dp, batch)
print_metrics(final_metrics_dp , title="dp")
