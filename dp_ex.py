import os
import functools
from typing import Any, Callable, Dict, Sequence, Tuple, NamedTuple

# Suppress verbose logging from libraries
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from absl import logging

logging.set_verbosity(logging.WARNING)


def sim_multiCPU_dev(num_devices=8):
    """Simulates multiple CPU devices for parallelism demonstrations."""
    os.environ.update(
        {
            "XLA_FLAGS": f"--xla_force_host_platform_device_count={num_devices}",
            "JAX_PLATFORMS": "cpu",
        }
    )
    print(f"✅ Simulated {num_devices} CPU devices.")


sim_multiCPU_dev()

import flax.linen as nn
from flax.training import train_state
import jax
import jax.numpy as jnp
import numpy as np
import optax

from jax import lax
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from ml_collections import ConfigDict

PyTree = Any
Parameter = jax.Array | nn.Partitioned
Metrics = Dict[str, Tuple[jax.Array, ...]]




class Batch(NamedTuple):
    """Container for a batch of data."""

    inputs: jax.Array
    labels: jax.Array


class TrainState(train_state.TrainState):
    """A custom TrainState that also holds the PRNG key."""

    rng: jax.random.PRNGKey


def accum_grads(
    state: TrainState,
    batch: Batch,
    rng: jax.random.PRNGKey,
    num_minibatches: int,
    loss_fn: Callable,
) -> Tuple[PyTree, Metrics]:
    """Accumulates gradients over multiple smaller minibatches."""
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    def step_fn(i, args):
        (rng, cum_grads, cum_metrics) = args
        step_rng, rng = jax.random.split(rng)
        (loss, step_metrics), grads = grad_fn(
            state.params, state.apply_fn, batch, step_rng
        )
        cum_grads = jax.tree_util.tree_map(jnp.add, cum_grads, grads)
        cum_metrics = jax.tree_util.tree_map(jnp.add, cum_metrics, step_metrics)
        return rng, cum_grads, cum_metrics

    zero_grads = jax.tree_util.tree_map(jnp.zeros_like, state.params)
    zero_metrics = {
        "loss": (jnp.array(0.0), jnp.array(0)),
        "accuracy": (jnp.array(0), jnp.array(0)),
    }

    # Loop over minibatches
    rng, final_grads, final_metrics = lax.fori_loop(
        0, num_minibatches, step_fn, (rng, zero_grads, zero_metrics)
    )

    # Average the accumulated gradients and metrics
    final_grads = jax.tree_util.tree_map(lambda g: g / num_minibatches, final_grads)
    return final_grads, final_metrics


def print_metrics(metrics: Metrics, step: int, title: str):
    """Prints metrics in a readable format."""
    loss = metrics["loss"][0] / metrics["loss"][1]
    accuracy = metrics["accuracy"][0] / metrics["accuracy"][1]
    print(
        f"[{title.upper()}] Step {step}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}"
    )




def fold_rng_over_axis(rng: jax.random.PRNGKey, axis_name: str):
    """Ensures each device gets a different RNG key for operations like dropout."""
    axis_index = jax.lax.axis_index(axis_name)
    return jax.random.fold_in(rng, axis_index)


def get_default_config():
    """Returns the base configuration for the experiment."""
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
            model=MODEL_CONFIG,
            optimizer=OPTIMIZER_CONFIG,
            data=DATA_CONFIG,
            data_axis_name=MODEL_CONFIG.data_axis_name,
            seed=42,
            num_steps=10,
        )
    )
    return CONFIG


class Classifier(nn.Module):
    """A simple MLP classifier model."""

    config: ConfigDict

    @nn.compact
    def __call__(self, x: jax.Array, train: bool) -> jax.Array:
        x = nn.Dense(
            features=self.config.hidden_size,
            dtype=self.config.dtype,
            name="input_dense",
        )(x)
        x = nn.silu(x)
        x = nn.Dropout(rate=self.config.dropout_rate, deterministic=not train)(x)
        x = nn.Dense(
            features=self.config.num_classes,
            dtype=self.config.dtype,
            name="output_dense",
        )(x)
        return x.astype(jnp.float32)


def loss_fn(
    params: PyTree,
    apply_fn: Callable,
    batch: Batch,
    rng: jax.random.PRNGKey,
    config: ConfigDict,
):
    """Calculates loss and metrics for a single batch."""
    dropout_rng = fold_rng_over_axis(rng, config.data_axis_name)
    logits = apply_fn(
        {"params": params}, batch.inputs, train=True, rngs={"dropout": dropout_rng}
    )
    loss_vector = optax.softmax_cross_entropy_with_integer_labels(logits, batch.labels)
    correct_pred = jnp.equal(jnp.argmax(logits, axis=-1), batch.labels)
    bs = batch.inputs.shape[0]
    step_metrics = {
        "loss": (loss_vector.sum(), bs),
        "accuracy": (correct_pred.sum(), bs),
    }
    return loss_vector.mean(), step_metrics


# --- FSDP Helper Functions ---


@jax.named_scope("shard_params")
def shard_params(
    params: PyTree, axis_name: str, min_weight_size: int = 2**18
) -> PyTree:
    """Splits large parameter arrays along a designated axis for sharding."""
    # get axis indexs
    axis_idx = jax.lax.axis_index(axis_name)
    # find the total size of axis by adding 1 for each device in axis
    axis_size = jax.lax.psum(1, axis_name)

    def _split(x: Parameter) -> Parameter:
        value = x
        names = (None,) * value.ndim
        # condtion1 -> if the size is smaller than minimum weight size -> return
        if value.size <= min_weight_size:
            return x

        shape = value.shape
        # Shard along the largest possible axis , so argsort in descending order and fetch the last element
        idx = np.argsort(shape)[::-1]
        for i in idx:
            # if its fully divisible by axis_size
            if shape[i] % axis_size == 0:
                split_size = shape[i] // axis_size
                p_sharded = nn.Partitioned(
                    value=lax.dynamic_slice_in_dim(
                        value, axis_idx * split_size, split_size, axis=i
                    ),
                    names=names[:i] + (axis_name,) + names[i + 1 :],
                )
                return p_sharded
        return x

    return jax.tree_util.tree_map(_split, params)


def gather_array_with_mean_grads(x: jax.Array, axis: int, axis_name: str):
    """Gathers an array and defines a custom gradient for the backward pass."""
    axis_size = jax.lax.psum(1, axis_name)

    @jax.custom_gradient
    def f(x_shard):
        # The forward pass gathers the sharded array into a full one.
        # gather all shards into 1 array -> each device sees this full array
        x_full = jax.lax.all_gather(x_shard, axis_name, axis=axis, tiled=True)

        def grad_fn(g_full):
            # The backward pass scatters and averages the gradients.
            # first scatter back all the gradients in theirt respective devices then divide by total number of devices
            g_shard = (
                jax.lax.psum_scatter(
                    g_full, axis_name, scatter_dimension=axis, tiled=True
                )
                / axis_size
            )
            return (g_shard,)

        return x_full, grad_fn

    return f(x)


@jax.named_scope("gather_params")
def gather_params(params: PyTree, axis_name: str) -> PyTree:
    """Gathers sharded parameters into full parameters for computation."""

    def _gather(p: Parameter) -> Parameter:
        if isinstance(p, nn.Partitioned) and axis_name in p.names:
            shard_axis = p.names.index(axis_name)
            return gather_array_with_mean_grads(
                p.value, axis=shard_axis, axis_name=axis_name
            )
        return p

    return jax.tree_util.tree_map(
        _gather, params, is_leaf=lambda x: isinstance(x, nn.Partitioned)
    )


def shard_module_params(
    target: nn.Module | Callable, axis_name: str, min_weight_size: int = 2**18
) -> nn.Module | Callable:
    """
    A wrapper that applies sharding and gathering logic to a Flax model.
    It automatically uses gather_params function in forward pass and
    shard_params during storing back of parameters
    """
    return nn.map_variables(
        target,
        "params",
        trans_in_fn=functools.partial(gather_params, axis_name=axis_name),
        trans_out_fn=functools.partial(
            shard_params, axis_name=axis_name, min_weight_size=min_weight_size
        ),
        mutable=True,
    )


# --- FSDP Training Step ---


def train_step_fsdp(
    state: TrainState, metrics: Metrics, batch: Batch, config: ConfigDict
) -> Tuple[TrainState, Metrics]:
    """A single training step optimized for FSDP."""
    rng, step_rng = jax.random.split(state.rng)

    # Curry the loss function with config
    bound_loss_fn = functools.partial(loss_fn, config=config)

    # Grads are calculated on gathered params, but returned sharded
    grads, step_metrics = accum_grads(
        state, batch, step_rng, config.optimizer.num_minibatches, loss_fn=bound_loss_fn
    )

    # Sync gradients across the data-parallel dimension
    with jax.named_scope("sync_grads"):
        # No pmean needed here because the custom gradient in `gather` handles it
        pass

    new_state = state.apply_gradients(grads=grads, rng=rng)

    with jax.named_scope("sync_metrics"):
        step_metrics = jax.tree_util.tree_map(
            lambda x: jax.lax.psum(x, axis_name=config.data_axis_name), step_metrics
        )

    # Accumulate metrics
    if metrics is None:
        metrics = step_metrics
    else:
        metrics = jax.tree_util.tree_map(jnp.add, metrics, step_metrics)

    return new_state, metrics


# --- Main Execution ---

if __name__ == "__main__":
    print("\n🚀 Starting FSDP Training Demo 🚀")

    # 1. Configuration and Setup
    CONFIG = get_default_config()
    devices = np.array(jax.devices())
    mesh = Mesh(devices, (CONFIG.data_axis_name,))
    print(f"Using mesh: {mesh}")

    # 2. Create the FSDP Model
    # The key step: wrap the original Classifier with the sharding logic
    ShardedClassifier = shard_module_params(
        Classifier, axis_name=CONFIG.data_axis_name  # Pass the class itself
    )

    # 2. Then, create an instance of the new sharded class
    model_fsdp = ShardedClassifier(config=CONFIG.model)
    optimizer = optax.adamw(learning_rate=CONFIG.optimizer.learning_rate)

    # 3. Create Synthetic Data
    rng = jax.random.PRNGKey(CONFIG.seed)
    model_init_rng, data_rng = jax.random.split(rng)

    data_inp_rng, data_label_rng = jax.random.split(data_rng)
    batch = Batch(
        inputs=jax.random.normal(
            data_inp_rng, (CONFIG.data.batch_size, CONFIG.data.input_size)
        ),
        labels=jax.random.randint(
            data_label_rng, (CONFIG.data.batch_size,), 0, CONFIG.data.num_classes
        ),
    )
    print(f"Created synthetic batch with input shape: {batch.inputs.shape}")

    # 4. Initialization Function for FSDP
    def init_fsdp(rng_key, x, model, optimizer_fn) -> TrainState:
        """Initializes the sharded model state."""
        init_rng, state_rng = jax.random.split(rng_key)
        # The `shard_module_params` wrapper handles sharding during init
        variables = model.init({"params": init_rng}, x, train=False)
        params = variables.pop("params")
        return TrainState.create(
            apply_fn=model.apply, params=params, tx=optimizer_fn, rng=state_rng
        )

    # JIT and Shard Map the functions to replicate the model in each device
    init_fsdp_fn = jax.jit(
        shard_map(
            functools.partial(init_fsdp, model=model_fsdp, optimizer_fn=optimizer),
            mesh,
            in_specs=(
                P(),
                P(CONFIG.data_axis_name),
            ),  # we wanna shard the x (input) not the rng_key
            out_specs=P(),
            check_rep=False,
        )
    )

    #
    train_step_fsdp_fn = jax.jit(
        shard_map(
            functools.partial(train_step_fsdp, config=CONFIG),
            mesh,
            in_specs=(P(), P(), P(CONFIG.data_axis_name)),  # we wanna shard batch only
            out_specs=(
                P(),
                P(),
            ),  # state and metric should be replicated across devices
            check_rep=False,
        ),
        donate_argnames=("state", "metrics"),
    )

    # 6. Initialize State and Metrics
    state_fsdp = init_fsdp_fn(model_init_rng, batch.inputs)
    # Check sharding of a large parameter
    param_shape = jax.tree_util.tree_leaves(state_fsdp.params)[0].shape
    print(f"Shape of the first sharded parameter on one device: {param_shape}")

    # Create an empty container for metrics
    _, metrics_shape = jax.eval_shape(train_step_fsdp_fn, state_fsdp, None, batch)
    metrics_fsdp = jax.tree_util.tree_map(
        lambda x: jnp.zeros(x.shape, dtype=x.dtype), metrics_shape
    )

    # 7. Main Training Loop
    print("\n--- Starting Training Loop ---")
    for step in range(CONFIG.num_steps):
        # Reset metrics for each step to see per-step performance
        step_metrics = jax.tree_util.tree_map(
            lambda x: jnp.zeros(x.shape, dtype=x.dtype), metrics_shape
        )
        state_fsdp, step_metrics = train_step_fsdp_fn(state_fsdp, step_metrics, batch)

        # Metrics are already summed across devices and replicated.
        # They are scalars, so we can't index them with `[0]`.
        # We can pass them directly to the print function.
        print_metrics(step_metrics, step + 1, title="FSDP")

    print("\n🎉 Training finished successfully! 🎉")
