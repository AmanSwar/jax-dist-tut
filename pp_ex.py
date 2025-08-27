import functools
import jax.experimental
import jax.experimental.shard_map
import numpy as np
from pprint import pprint
from typing import Any, Callable, Dict, Tuple
from dataclasses import dataclass

# JAX/Flax/Optax imports
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState as FlaxTrainState
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from jax.tree_util import tree_map
from ml_collections import ConfigDict

# PyTree type alias for type hinting
PyTree = Any

# ==============================================================================
# ## 1. Utility Functions (Previously in external files)
# These functions are necessary for the script to run.
# ==============================================================================


def sim_multiCPU_dev(num_devices=8):
    """Simulates multiple CPU devices for parallelism testing."""
    # To run on a CPU, we can simulate multiple devices.
    # This flag needs to be set before JAX initializes its backends.
    jax.config.update("jax_cpu_devices", num_devices)
    print(f"Simulating {len(jax.devices())} CPU devices.")


@dataclass
class Batch:
    """A simple dataclass to hold a batch of training data."""

    inputs: jax.Array
    labels: jax.Array


class TrainState(FlaxTrainState):
    """A custom TrainState to include the PRNG key."""

    rng: jax.Array


def fold_rng_over_axis(rng: jax.Array, axis: str | Tuple[str, ...]) -> jax.Array:
    """Folds the RNG key over the given mesh axes to get a unique key per device."""
    if isinstance(axis, str):
        axis = (axis,)
    for axis_name in axis:
        rng = jax.random.fold_in(rng, jax.lax.axis_index(axis_name))
    return rng


def sync_gradients(grads: PyTree, axis_names: Tuple[str, ...]) -> PyTree:
    """Averages gradients across all specified axes in the mesh."""
    return tree_map(lambda x: jax.lax.pmean(x, axis_name=axis_names), grads)


def accumulate_gradients(
    state: TrainState,
    batch: Batch,
    rng: jax.Array,
    num_minibatches: int,
    loss_fn: Callable,
) -> Tuple[PyTree, Dict[str, Any]]:
    """Accumulates gradients over multiple minibatches."""
    batch_size = batch.inputs.shape[0]
    minibatch_size = batch_size // num_minibatches

    # Reshape data into minibatches
    minibatches = tree_map(
        lambda x: x.reshape((num_minibatches, minibatch_size) + x.shape[1:]), batch
    )

    # Define the gradient function for a single minibatch
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    def compute_grad(params, minibatch, rng):
        (_, metrics), grads = grad_fn(params, state.apply_fn, minibatch, rng)
        return grads, metrics

    # Vmap over the minibatches to compute gradients for each
    vmapped_grad_fn = jax.vmap(
        functools.partial(compute_grad, state.params), in_axes=(0, 0), out_axes=0
    )

    rngs = jax.random.split(rng, num_minibatches)
    grads, metrics = vmapped_grad_fn(minibatches, rngs)

    # Average the gradients and metrics across all minibatches
    avg_grads = tree_map(lambda x: jnp.mean(x, axis=0), grads)
    avg_metrics = tree_map(lambda x: jnp.sum(x, axis=0), metrics)

    return avg_grads, avg_metrics


def print_metrics(metrics: Dict[str, Tuple[jax.Array, ...]], step: int):
    """Computes and prints metrics like loss and accuracy."""
    loss = metrics["loss"][0] / metrics["loss"][1]
    accuracy = metrics["accuracy"][0] / metrics["accuracy"][1] * 100
    print(f"Step {step:03d} | Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%")


def get_num_params(params: PyTree) -> int:
    """Calculates the total number of parameters in a PyTree."""
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


# ==============================================================================
# ## 2. Model and Pipeline Logic (User's Original Code)
# ==============================================================================

Parameter = jax.Array | nn.Partitioned
Metrics = Dict[str, Tuple[jax.Array, ...]]


class MLPBlock(nn.Module):
    config: ConfigDict
    train: bool

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        input_feat = x.shape[-1]
        residual = x
        x = nn.LayerNorm(dtype=self.config.dtype, name="pre_norm")(x)
        x = nn.Dense(
            features=self.config.hidden_size * self.config.mlp_expansion,   
            dtype=self.config.dtype,
            name="input_dense",
        )(x)
        x = nn.silu(x)
        x = nn.Dropout(rate=self.config.dropout_rate, deterministic=not self.train)(x)
        x = nn.Dense(features=input_feat, dtype=self.config.dtype, name="output")(x)
        return x + residual


class MLPLayers(nn.Module):
    config: ConfigDict
    train: bool

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        block_class = MLPBlock
        if "MLP" in self.config.remat:
            block_class = nn.remat(block_class, prevent_cse=False)

        # This is the module whose parameters will be replicated `num_layers` times
        block = block_class(config=self.config, train=self.train, name="block")

        # nn.scan applies the same module (`block`) multiple times
        x, _ = nn.scan(
            lambda module, carry, _: (module(carry), ()),
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            length=self.config.num_layers,
        )(block, x, ())
        return x


def stack_params(
    params: PyTree,
    axis_name: str,
    axis: int = 0,
    mask_except: jax.Array | int | None = None,
) -> PyTree:
    """Stacks sharded params along a given axis."""

    def _stack(x):
        if isinstance(x, nn.Partitioned):
            value, names = x.value, x.names
        else:
            value, names = x, (None,) * x.ndim

        if mask_except is not None:
            axis_index = jax.lax.axis_index(axis_name)
            value = jnp.where(axis_index == mask_except, value, jnp.zeros_like(value))

        value = jnp.expand_dims(value, axis)
        names = names[:axis] + (axis_name,) + names[axis:]

        return nn.Partitioned(value, names=names)

    return tree_map(_stack, params, is_leaf=lambda x: not isinstance(x, nn.Partitioned))


def unstack_params(params: PyTree, axis_name: str):
    """Unstacks params along a given axis."""

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

    return tree_map(_unstack, params, is_leaf=lambda x: isinstance(x, nn.Partitioned))


def execute_pipeline_step(
    model: nn.Module,
    state: jax.Array,
    input: jax.Array,
    *args,
    model_axis_name: str,
    **kwargs,
) -> Tuple[jax.Array, jax.Array]:
    """Executes a single step of a microbatch through one pipeline stage."""
    #get total number of stages
    num_stages = jax.lax.psum(1, model_axis_name)
    #stage index -> depending on device
    stage_index = jax.lax.axis_index(model_axis_name)

    # The first stage takes the original input; other stages take the output from the previous one.
    state = jnp.where(stage_index == 0, input, state)
    state = model(state, *args, **kwargs)

    # Only the last stage produces the final output.
    output = jnp.where(stage_index == num_stages - 1, state, jnp.zeros_like(state))

    # send the state to the next device in the pipeline for the next step.
    state = jax.lax.ppermute(
        state,
        model_axis_name,
        perm=[(i, (i + 1) % num_stages) for i in range(num_stages)],
    )
    return (state, output)


@jax.named_scope("pipeline")
def execute_pipeline(
    module: nn.Module,
    x: jax.Array,
    *args,
    num_microbatches: int,
    model_axis_name: str,
    **kwargs,
) -> jax.Array:
    """Executes the full pipeline schedule for a batch."""
    #total devices
    num_stages = jax.lax.psum(1, model_axis_name)
    batch_size = x.shape[0]
    microbatch_size = batch_size // num_microbatches
    #reshape from (batch_size , dims...) -> (num_microbatches , microbatch_size , dims...)
    microbatches = jnp.reshape(x, (num_microbatches, microbatch_size, *x.shape[1:]))

    # Add "bubble" inputs to keep the pipeline full.
    inputs = jnp.concatenate(
        [
            microbatches,
            jnp.zeros(
                (num_stages - 1, *microbatches.shape[1:]), dtype=microbatches.dtype
            ),
        ],
        axis=0,
    )
    # Initial state is zeros.
    state = jnp.zeros_like(microbatches[0])

    n_iter = inputs.shape[0]

    # Scan over the pipeline steps.
    _, outputs = nn.scan(
        functools.partial(
            execute_pipeline_step, *args, model_axis_name=model_axis_name, **kwargs
        ),
        variable_broadcast={"params": True},
        split_rngs={"params": False, "dropout": True},
        length=n_iter,
        in_axes=0,
        out_axes=0,
    )(module, state, inputs)

    # Concatenate the final outputs from the last stage.
    outputs = jnp.concatenate(outputs[-num_microbatches:], axis=0)
    return outputs


class PipelineModule(nn.Module):
    model_axis_name: str
    num_microbatches: int
    module_fn: Callable[..., nn.Module]

    @nn.compact
    def __call__(self, *args, **kwargs):
        module = self.module_fn()
        return execute_pipeline(
            module,
            *args,
            **kwargs,
            num_microbatches=self.num_microbatches,
            model_axis_name=self.model_axis_name,
        )


class ModelParallelWrapper(nn.Module):
    model_axis_name: str
    module_fn: Callable[..., nn.Module]
    mask_except_model_idx: int | None = None
    split_rngs: bool = True
    module_kwargs: FrozenDict[str, Any] = FrozenDict({})

    @nn.compact
    def __call__(self, *args, **kwargs):
        if self.is_initializing() and self.split_rngs:
            self.scope.rngs["params"] = self.scope.rngs["params"].replace(
                rng=fold_rng_over_axis(
                    self.scope.rngs["params"].rng, self.model_axis_name
                )
            )

        module = nn.map_variables(
            target=functools.partial(
                self.module_fn,
                name="sharded",
                **self.module_kwargs,
            ),
            trans_in_fn=functools.partial(
                unstack_params, axis_name=self.model_axis_name
            ),
            trans_out_fn=functools.partial(
                stack_params,
                axis_name=self.model_axis_name,
                mask_except=self.mask_except_model_idx,
            ),
            mapped_collections=["params"],
            mutable=True,
        )()
        return module(*args, **kwargs)


class PPClassifier(nn.Module):
    config: ConfigDict
    pipeline_module_class: Callable[..., nn.Module] = PipelineModule

    @nn.compact
    def __call__(self, x: jax.Array, train: bool) -> jax.Array:
        # Input layer: Only exists on the first device.
        x = ModelParallelWrapper(
            module_fn=functools.partial(
                nn.Dense,
                features=self.config.hidden_size,
                dtype=self.config.dtype,
            ),
            split_rngs=True,
            model_axis_name=self.config.model_axis_name,
            mask_except_model_idx=0,
            name="input_dense",
        )(x)

        # Main pipeline stages (MLP layers)
        stage_module_fn = functools.partial(
            MLPLayers, config=self.config, train=train, name="mlp_layers"
        )
        pipeline_module_fn = functools.partial(
            self.pipeline_module_class,
            model_axis_name=self.config.model_axis_name,
            num_microbatches=self.config.num_microbatches,
            module_fn=stage_module_fn,
        )
        module = ModelParallelWrapper(
            module_fn=pipeline_module_fn,
            model_axis_name=self.config.model_axis_name,
            name="pipeline",
        )
        x = module(x)

        # Output layers: Only exist on the last device.
        output_wrapper = functools.partial(
            ModelParallelWrapper,
            model_axis_name=self.config.model_axis_name,
            mask_except_model_idx=self.config.model_axis_size - 1,
        )
        x = output_wrapper(
            module_fn=functools.partial(nn.LayerNorm, dtype=self.config.dtype),
            name="output_norm",
        )(x)
        x = output_wrapper(
            module_fn=functools.partial(
                nn.Dense, features=self.config.num_classes, dtype=self.config.dtype
            ),
            name="output_dense",
        )(x)
        x = x.astype(jnp.float32)
        return x


def get_default_classifier_config() -> ConfigDict:
    data_config = ConfigDict(dict(batch_size=128, num_classes=10, input_size=784))
    model_config = ConfigDict(
        dict(
            hidden_size=512,
            mlp_expansion=1,
            dropout_rate=0.1,
            num_layers=8,  # Total layers in the model
            dtype=jnp.float32,
            num_classes=data_config.num_classes,
            remat=(),
            data_axis_name="data",
            model_axis_name="model",
            model_axis_size=4,  # Number of pipeline stages
            num_microbatches=8,
        )
    )
    # Layers per stage
    model_config.num_layers //= model_config.model_axis_size
    optimizer_config = ConfigDict(dict(learning_rate=1e-3, num_minibatches=1))
    config = ConfigDict(
        dict(
            model=model_config,
            optimizer=optimizer_config,
            data=data_config,
            data_axis_name=model_config.data_axis_name,
            model_axis_name=model_config.model_axis_name,
            model_axis_size=model_config.model_axis_size,
            seed=42,
        )
    )
    return config


def loss_fn(
    params: PyTree, apply_fn: Any, batch: Batch, rng: jax.Array
) -> Tuple[jax.Array, Dict[str, Any]]:
    dropout_rng = fold_rng_over_axis(
        rng, (config.data_axis_name, config.model_axis_name)
    )
    logits = apply_fn(
        {"params": params}, batch.inputs, train=True, rngs={"dropout": dropout_rng}
    )
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch.labels)
    correct_pred = jnp.equal(jnp.argmax(logits, axis=-1), batch.labels)

    # Mask out loss and accuracy for all pipeline stages except the last one.
    model_idx = jax.lax.axis_index(config.model_axis_name)
    model_size = jax.lax.psum(1, config.model_axis_name)
    is_last_stage = model_idx == model_size - 1

    loss = jnp.where(is_last_stage, loss, 0.0)
    correct_pred = jnp.where(is_last_stage, correct_pred, False)
    effective_batch_size = jnp.where(is_last_stage, batch.inputs.shape[0], 0)

    step_metrics = {
        "loss": (loss.sum(), effective_batch_size),
        "accuracy": (correct_pred.sum(), effective_batch_size),
    }
    return loss.mean(), step_metrics


def train_step_pp(
    state: TrainState, metrics: Metrics | None, batch: Batch
) -> Tuple[TrainState, Metrics]:
    rng, step_rng = jax.random.split(state.rng)
    grads, step_metrics = accumulate_gradients(
        state, batch, step_rng, config.optimizer.num_minibatches, loss_fn=loss_fn
    )
    # Sync gradients across both data and model parallel axes.
    with jax.named_scope("sync_gradients"):
        grads = sync_gradients(grads, (config.data_axis_name, config.model_axis_name))

    new_state = state.apply_gradients(grads=grads, rng=rng)

    # Sum metrics across all replicas.
    with jax.named_scope("sync_metrics"):
        step_metrics = tree_map(
            lambda x: jax.lax.psum(
                x, axis_name=(config.data_axis_name, config.model_axis_name)
            ),
            step_metrics,
        )

    if metrics is None:
        metrics = step_metrics
    else:
        metrics = tree_map(jnp.add, metrics, step_metrics)

    return new_state, metrics


def train_pipeline_model(
    config: ConfigDict,
    mesh: Mesh,
    batch: Batch,
    model_init_rng: jax.Array,
    num_steps: int,
) -> TrainState:

    model_pp = PPClassifier(config=config.model)
    optimizer = optax.adamw(learning_rate=config.optimizer.learning_rate)

    def init_fn(rng: jax.random.PRNGKey, x: jax.Array, model: nn.Module) -> TrainState:
        init_rng, rng = jax.random.split(rng)
        variables = model.init({"params": init_rng}, x, train=False)
        params = variables.pop("params")
        return TrainState.create(
            apply_fn=model.apply, params=params, tx=optimizer, rng=rng
        )

    # Shard the initialization function across the mesh
    init_pp_fn = shard_map(
        functools.partial(init_fn, model=model_pp),
        mesh,
        in_specs=(P(), P(config.data_axis_name)),
        out_specs=P(),
        check_rep=False,
    )
    state_pp_shapes = jax.eval_shape(init_pp_fn, model_init_rng, batch.inputs)
    state_pp_specs = nn.get_partition_spec(state_pp_shapes)

    # JIT the sharded initialization
    jit_init_fn = jax.jit(
        shard_map(
            functools.partial(init_fn, model=model_pp),
            mesh,
            in_specs=(P(), P(config.data_axis_name)),
            out_specs=state_pp_specs,
            check_rep=False,
        )
    )
    state_pp = jit_init_fn(model_init_rng, batch.inputs)

    print(f"Model initialized with {get_num_params(state_pp.params):,} parameters.")
    # This pprint call is moved here, after state_pp is initialized.
    print("Shape of parameters in one MLP block per pipeline stage:")
    pprint(
        tree_map(
            lambda x: x.shape,
            state_pp.params["pipeline"]["sharded"]["mlp_layers"]["block"],
        )
    )

    # JIT the sharded training step
    train_step_pp_fn = jax.jit(
        shard_map(
            train_step_pp,
            mesh,
            in_specs=(state_pp_specs, P(), P(config.data_axis_name)),
            out_specs=(state_pp_specs, P()),
            check_rep=False,
        ),
        donate_argnames=("state", "metrics"),
    )

    # Initialize metrics accumulator
    _, metric_shapes = jax.eval_shape(train_step_pp_fn, state_pp, None, batch)
    metrics_pp = tree_map(lambda x: jnp.zeros(x.shape, dtype=x.dtype), metric_shapes)

    print("\n--- Starting Training ---")
    for step in range(1, num_steps + 1):
        state_pp, metrics_pp = train_step_pp_fn(state_pp, metrics_pp, batch)
        # We print metrics every step for this example
        print_metrics(metrics_pp, step)
        metrics_pp = tree_map(jnp.zeros_like, metrics_pp)  # Reset metrics

    return state_pp


# ==============================================================================
# ## 3. Main Execution Block
# ==============================================================================

if __name__ == "__main__":
    """
    This block sets up the configuration, devices, data, and runs the training loop.
    """
    # 1. Setup simulated devices
    # We need 8 devices for a 2x4 mesh (data_axis x model_axis)
    sim_multiCPU_dev(num_devices=8)

    # 2. Configuration
    num_train_steps = 10
    config = get_default_classifier_config()

    # Ensure the mesh shape matches the configuration
    data_axis_size = len(jax.devices()) // config.model.model_axis_size

    # 3. Create the device mesh
    device_array = np.array(jax.devices()).reshape(
        data_axis_size, config.model.model_axis_size
    )
    mesh = Mesh(device_array, (config.data_axis_name, config.model_axis_name))
    print(f"Created a {data_axis_size}x{config.model_axis_size} device mesh.")

    # 4. Create toy data
    rng = jax.random.PRNGKey(config.seed)
    model_init_rng, data_rng = jax.random.split(rng, 2)
    data_inputs_rng, data_labels_rng = jax.random.split(data_rng, 2)

    batch = Batch(
        inputs=jax.random.normal(
            data_inputs_rng, (config.data.batch_size, config.data.input_size)
        ),
        labels=jax.random.randint(
            data_labels_rng, (config.data.batch_size,), 0, config.data.num_classes
        ),
    )

    # 5. Run the training
    final_state = train_pipeline_model(
        config=config,
        mesh=mesh,
        batch=batch,
        model_init_rng=model_init_rng,
        num_steps=num_train_steps,
    )

    print("\n--- Training Finished ---")
