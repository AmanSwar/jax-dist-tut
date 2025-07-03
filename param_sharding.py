from util import sim_multiCPU_dev

# simulate multiple devices on CPU
sim_multiCPU_dev()


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

from util import Pytree, TrainState, Batch, Metrics, accum_grads, print_metrics

# configurations to be used


# data config
DATA_CONFIG = ConfigDict(
    dict(
        batch_size=128,
        num_classes=10,
        input_size=784,
    )
)

# model config
MODEL_CONFIG = ConfigDict(
    dict(
        hidden_size=512,
        dropout_rate=0.1,
        dtype=jnp.bfloat16,
        num_classes=DATA_CONFIG.num_classes,
        data_axis_name="data",
        lr=1e-4,
    )
)

# main config obj
config = ConfigDict(
    dict(model=MODEL_CONFIG, data=DATA_CONFIG, seed=6969, num_minibatches=4)
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
        # Conditions where I don't have to shard the params
        # - when its already paritioned
        # - if its less than min_size
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
            # we are going to shared alonog the largest dim first
            shape = value.shape
            idx = np.argsort(shape)[::-1]  # shape in descending order of largest axis

            # iterate over every dim to check
            for i in idx:
                # check condition -> can the dim be equally partitioned across different devices or not
                if shape[i] % axis_size == 0 and names[i] is None:

                    split_size = shape[i] // axis_size
                    p_sharded = nn.Partitioned(
                        value=lax.dynamic_slice_in_dim(
                            value, axis_idx * split_size, split_size, axis=i
                        ),
                        # add axis name in the current index
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


# now a function which cummulates all the value -> finds gradient -> distribute the gradients
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


# launch function for grather_arr_mean_grads
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


# function for applying gather and sharding ops for each param
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


def init_model(rng, x: jax.Array, model: nn.Module) -> TrainState:
    init_rng, rng = jax.random.split(rng)
    var = model.init({"params": init_rng}, x, train=False)
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


config.model.min_weight_size = (
    2**4
)  # set to smaller one cuz this is a experimental script

model_fsdp = Classifier(config=config.model)
devices = np.array(jax.devices())
# init mesh
mesh = Mesh(devices, (config.data_axis_name,))

init_fsdp_fn = shard_map(
    functools.partial(init_model, model=model_fsdp),
    mesh,
    in_specs=(P(), P(config.model.data_axis_name)),
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


# function to handle the grads of the axes which are not replicated
# but have different grads across devices -> average over it
def sync_gradients(
    grads: Pytree,
    axis_names: Sequence[str],
) -> Pytree:

    def _sync_grad(g):

        if isinstance(g, nn.Partitioned):

            repl_axis_names = [
                name
                for name in axis_names
                if name not in jax.tree_util.tree_leaves(g.names)
            ]

            if len(repl_axis_names) == 0:
                return g

            else:
                return g.replace(
                    value=jax.lax.pmean(g.value, axis_name=repl_axis_names)
                )

        else:

            return jax.lax.pmean(g, axis_name=axis_names)

    return jax.tree_util.tree_map(
        _sync_grad, grads, is_leaf=lambda x: isinstance(x, nn.Partitioned)
    )


def loss_fn(params, apply_fn, batch, rng):

    dropout_rng = fold_rng_over_axis(rng, CONFIG.data_axis_name)
    logits = apply_fn(
        {"params": params}, batch.inputs, train=True, rngs={"dropout": dropout_rng}
    )

    loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch.labels)

    correct_pred = jnp.equal(jnp.argmax(logits, axis=-1), batch.labels)

    bs = batch.inputs.shape[0]
    step_metrics = {"loss": (loss.sum(), bs), "accuracy": (correct_pred.sum(), bs)}
    loss = loss.mean()

    return loss, step_metrics


def train_step_fsdp(state: TrainState, metrics: Metrics, batch: Batch):
    rng, step_rng = jax.random.split(state.rng)

    grads, step_metrics = accum_grads(
        state, batch, step_rng, config.num_minibatches, loss_fn=loss_fn
    )

    with jax.named_scope("sync_grad"):
        grads = sync_gradients(grads, (config.data_axis_name,))

    new_state = state.apply_gradients(grads=grads, rng=rng)

    with jax.named_scope("synch_metrics"):

        step_metrics = jax.tree_util.tree_map(
            lambda x: jax.lax.psum(x, axis_name=config.data_axis_name), step_metrics
        )

    if metrics is None:
        metrics = step_metrics

    else:
        metrics = jax.tree_util.tree_map(jnp.add, metrics, step_metrics)

    return new_state, metrics


train_step_fsdp_fn = jax.jit(
    shard_map(
        train_step_fsdp,
        mesh,
        in_specs=(state_fsdp_specs, P(), P(config.data_axis_names)),
        out_specs=(state_fsdp_specs, P()),
        check_rep=False,
    ),
    donate_argnames=("state", "metrics"),
)
_, metric_shapes = jax.eval_shape(
    train_step_fsdp_fn,
    state_fsdp,
    None,
    batch,
)
metrics_fsdp = jax.tree_util.tree_map(
    lambda x: jnp.zeros(x.shape, dtype=x.dtype), metric_shapes
)
for _ in range(15):
    state_fsdp, metrics_fsdp = train_step_fsdp_fn(state_fsdp, metrics_fsdp, batch)
final_metrics_fsdp = jax.tree_util.tree_map(
    lambda x: jnp.zeros(x.shape, dtype=x.dtype), metric_shapes
)
state_fsdp, final_metrics_fsdp = train_step_fsdp_fn(
    state_fsdp, final_metrics_fsdp, batch
)
print_metrics(final_metrics_fsdp, "FSDP - Final metrics")
