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

from util import Pytree , TrainState , Batch

MODEL_CONFIG = ConfigDict(
    dict(
        hidden_size=512,
        dropout_rate=0.1,
        dtype=jnp.bfloat16,
        num_classes=DATA_CONFIG.num_classes,
        data_axis_name="data",
        lr=1e-4
    )
)

DATA_CONFIG = ConfigDict(
    dict(
        batch_size=128,
        num_classes=10,
        input_size=784,
    )
)

config = ConfigDict(
    dict(
        model = MODEL_CONFIG,
        data = DATA_CONFIG,
        seed=6969,
    )
)


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


def gather_arr_mean_grads(x: jax.Array, axis: int, axis_name: str):
    axis_size = jax.lax.psum(1, axis_name)

    @jax.custom_gradient
    def f(x):
        def grad_fn(g):
            return (
                jax.lax.psum_scatter(g, axis_name, scatter_dimension=axis, tiled=True)
                / axis_size
            )

        return jax.lax.all_gather(x, axis_name, axis=axis, tiled=True), grad_fn

    return f(x)


@jax.named_scope("gather_params")
def gather_params(params: Pytree, axis_name: str) -> Pytree:

    def _gather(p):

        if isinstance(p, nn.Partitioned) and axis_name in p.names:

            param_shard = p.names
            shared_axis = param_shard.index(axis_name)

            value = gather_arr_mean_grads(
                p.value, axis=shared_axis, axis_name=axis_name
            )

            param_shard = (
                param_shard[:shared_axis] + (None,) + param_shard[shared_axis + 1 :]
            )

            if any([name is not None for name in param_shard]):
                return nn.Partitioned(value, param_shard)

            else:
                return value

        else:
            return p

    return jax.tree_util.tree_map(
        _gather, params, is_leaf=lambda x: isinstance(x, nn.Partitioned)
    )


def shard_module_params(
    target: nn.Module | Callable, axis_name: str, min_weight_size: int = 2**18
) -> nn.Module | Callable:

    return nn.map_variables(
        target,
        trans_in_fn=functools.partial(gather_params, axis_name=axis_name),
        trans_out_fn=functools.partial(
            shard_params, axis_name=axis_name, min_weight_size=min_weight_size
        ),
        mapped_collections="params",
        mutable=True,
    )


class Classifier(nn.Module):

    config: ConfigDict

    @nn.compact
    def __call__(self, x: jax.Array, train: bool) -> jax.Array:

        sharded_dense = shard_module_params(
            nn.Dense,
            axis_name=self.config.model.data_axis_name,
            min_weight_size=self.config.min_weight_size,
        )

        x = sharded_dense(
            features=self.config.hidden_size,
            dtype=self.config.dtype,
            name="input_dense",
        )(x)

        x = nn.silu(x)
        x = nn.Dropout(rate=self.config.dropout_rate, deterministic=not train)(x)

        x = sharded_dense(
            features=self.config.num_classes,
            dtype=self.config.dtype,
            name="output_dense",
        )(x)

        x = x.astype(jnp.float32)

        return x


def init_model(
        rng,
        x : jax.Array,
        model : nn.Module
) -> TrainState:
    init_rng , rng = jax.random.split(rng)
    var = model.init({'params' : init_rng} , x , train=False)
    params = var.pop("params")

    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.adamw(
            learning_rate=config.model.lr,
        ),
        rng=rng,
    )

    return state


config.model.min_weight_size = 2**4  #set to smaller one cuz this is a experimental script

model_fsdp = Classifier(config=config.model)
devices = np.array(jax.devices())
# init mesh
mesh = Mesh(devices, (config.data_axis_name,))

init_fsdp_fn = shard_map(
    functools.partial(init_model , model=model_fsdp),
    mesh,
    in_specs=(P() , P(config.model.data_axis_name)),
    out_specs=P(),
    check_rep=False,
)

rng = jax.random.PRNGKey(config.seed)

model_init_rng, data_inp_rng, datal_label_rng = jax.random.split(rng, 3)

state_fsdp_shape = jax.eval_shape(init_fsdp_fn)
state_fsdp_specs = nn.get_partition_spec(state_fsdp_shape)

init_fsdp_fn = shard_map(
    functools.partial(init_model, model=model_fsdp),
    mesh,
    in_specs=(P(), P(config.model.data_axis_name)),
    out_specs=state_fsdp_specs,
    check_rep=False,
)

batch = Batch(
    inputs=jax.random.normal(
        data_inp_rng,
        (config.data.batch_size, config.data.input_size),  # type: ignore
    ),
    labels=jax.random.normal(
        datal_label_rng,
        (config.data.batch_size,),
        0,
        config.data.num_classes,
    ),
)
state_fsdp = init_fsdp_fn(model_init_rng, batch.inputs)
