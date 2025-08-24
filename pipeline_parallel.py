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


from util import (
    Batch,
    TrainState,
    accum_grads,
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
    """
    Stakcs sharded params along a givesn axis

    Args:
        params (PyTree): model parameters
        axis_name (str): name of the axis to stakc along
        mask_except (jax.Array | int | None): only the mask_except-th shard will be non zero
        axis (int, optional): index of the axis to stack along

    Returns:
        PyTree: Parameters 
    """


    def _stack(x):
        """
        Core stack logic - used on each leaf of a PyTree
        Args:
            x (jax.Array): input

        """

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
    """
    Unstack params along a given axis

    Args:
        params (PyTree) : params
        axis_name (str): axis along which to unstack
    """
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
        



def execute_pipeline_step(
        module : nn.Module,
        state : jax.Array,
        input : jax.Array,
        *args,
        model_axis_name : str,
        **kwargs
) -> Tuple[jax.Array , jax.Array]:
    """
    Single micro batch pipeline step

    Args:
        module (nn.Module): stage to be executed
        state (jax.Array): output of last stage
        input (jax.Array): original input
        model_axis_name (str): name of modle exis in the mesh

    Returns:
        Tuple[jax.Array , jax.Array]: _description_
    """
    
    #total no. of stage = total axis name
    num_stages = jax.lax.psum(1 , model_axis_name)
    # indexify the axis names
    stage_index = jax.lax.axis_index(model_axis_name)
    state = jnp.where(stage_index == 0 , input , state)
    state = module(state , *args , **kwargs)


    output = jnp.where(stage_index == num_stages -1 , state , jnp.zeros_like(state))

    state = jax.lax.ppermute(
        state,
        model_axis_name,
        perm= [(i , (i+1) % num_stages) for i in range(num_stages)]
    )

    return (state , output)



@jax.named_scope("pipeline")
def execute_pipeline(
    module : nn.Module,
    x : jax.Array,
    *args,
    num_microbatches : int,
    model_axis_name : str,
    **kwargs
) -> jax.Array:
    
    num_stages = jax.lax.psum(1 , model_axis_name)

    batch_size = x.shape[0]

    microbatch_size = batch_size // num_microbatches
    microbatches = jnp.reshape(x , (num_microbatches , microbatch_size , *x.shape[1:]))

    inputs = jnp.concatenate(
        [
            microbatches,
            jnp.zeros((num_stages -1 , *microbatches.shape[1:]), dtype=microbatches.dtype)
        ],
        axis=0
    )

    state = jnp.zeros_like(microbatches[0])

    n_iter = inputs.shape[0]

    _ , outputs = nn.scan(
        functools.partial(
            execute_pipeline_step,
            *args,
            model_axis_name=model_axis_name,
            **kwargs,
        ),
        variable_broadcast = {"params" : True},
        split_rngs = {"params" : False , "dropout" : True},
        length=n_iter,
        in_axes=0,
        out_axes=0,
    )(module , state , inputs)


    outputs = jnp.concatenate(outputs[-num_microbatches:] , axis=0)

    return outputs


class PipelineModule(nn.Module):

    model_axis_name : str
    num_microbatches : int
    module_fn : Callable[... , nn.Module]

    @nn.compact
    def __call__(
        self,
        *args,
        **kwargs
    ):
        
        module = self.module_fn()

        return execute_pipeline(
            module,
            *args,
            **kwargs,
            num_microbatches=self.num_microbatches,
            model_axis_name=self.model_axis_name,
        )
    
class ModelParallelWrapper(nn.Module):

    model_axis_name :str
    module_fn : Callable[... , nn.Module]
    mask_except_model_idx : int | None = None
    split_rngs : bool = True
    module_kwargs : FrozenDict[str , Any] = FrozenDict({})

    @nn.compact
    def __call__(self, *args , **kwargs):

        if self.is_initializing() and self.split_rngs:
            self.scope.rngs["params"] = self.scope.rngs["params"].replace(
                rng=fold_rng_over_axis(self.scope.rngs["params"].rng , self.model_axis_name)
            )

            module = nn.map_variables(
                target=functools.partial(
                    self.module_fn,
                    name="shareded",
                    **self.module_kwargs,
                ),

                trans_in_fn=functools.partial(unstack_params , axis_name=self.model_axis_name),
                trans_out_fn= functools.partial(
                    stack_params,
                    axis_name=self.model_axis_name,
                    mask_except=self.mask_except_model_idx,
                ),
                mapped_collections="params",
                mutable=True,
            )()

            return module(
                *args,
                **kwargs
            )
        

class PPClassifier(nn.Module):
    config : ConfigDict
    pipeline_module_class : Callable[... , nn.Module] = PipelineModule


    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        train : bool
    ) -> jax.Array:
        
        x = ModelParallelWrapper(
            module_fn=functools.partial(
                nn.Dense,
                features=self.config.hidden_size, # type: ignore
                dtype=self.config.dtype,
            ),
            split_rngs=True,
            model_axis_name=self.config.model_axis_name,
            mask_except_model_idx=0,
            name="input_dense",
        )(x)

        stage_module_fn = functools.partial(
            MLPLayers , config=self.config , train=train , name="mlp_layer"
        )


        pipeline_module_fn = functools.partial(
            self.pipeline_module_class,
            model_axis_name = self.config.model_axis_name,
            num_microbatches=self.config.num_microbatches,
            module_fn=stage_module_fn,
        )

        module = ModelParallelWrapper(
            module_fn=pipeline_module_fn,
            model_axis_name=self.config.model_axis_name,
            name="pipeline",
        )

        x = module(x)

        output_wrapper = functools.partial(
            ModelParallelWrapper,
            model_axis_name=self.config.model_axis_name,
            mask_except_model_idx=self.config.model_axis_size-1,
        )

        x = output_wrapper(
            module_fn=functools.partial(nn.LayerNorm , dtype=self.config.dtype),
            name="output_norm"
        )(x) # type: ignore

        x = output_wrapper(
            module_fn=functools.partial(
                nn.Dense, features=self.config.num_classes, dtype=self.config.dtype # type: ignore
            ),
            name="output_dense",
        )(x)

        x = x.astype(jnp.float32)
        return x
    


def get_default_classifier_config() -> ConfigDict:

    data_config = ConfigDict(
        dict(
            batch_size=128,
            num_classes=10,
            input_size=784,
        )
    )
    model_config = ConfigDict(
        dict(
            hidden_size=512,
            mlp_expansion=1,
            dropout_rate=0.1,
            num_layers=8,
            dtype=jnp.float32,
            num_classes=data_config.num_classes,
            remat=(),
            data_axis_name="data",
            model_axis_name="model",
            model_axis_size=4,
            num_microbatches=8,
        )
    )
    model_config.num_layers //= model_config.model_axis_size  # Layers distributed over model axis.
    optimizer_config = ConfigDict(
        dict(
            learning_rate=1e-3,
            num_minibatches=1,
        )
    )
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

device_array = np.array(jax.devices()).reshape(-1, config.model_axis_size)
mesh = Mesh(device_array, (config.data_axis_name, config.model_axis_name))

model_pp = PPClassifier(config=model_config)
optimizer = optax.adamw(
    learning_rate=config.optimizer.learning_rate,
)


rng = jax.random.PRNGKey(config.seed)
model_init_rng, data_inputs_rng, data_labels_rng = jax.random.split(rng, 3)
batch = Batch(
    inputs=jax.random.normal(data_inputs_rng, (config.data.batch_size, config.data.input_size)),
    labels=jax.random.randint(
        data_labels_rng, (config.data.batch_size,), 0, config.data.num_classes
    ),
)



            

def init_fn(rng: jax.random.PRNGKey, x: jax.Array, model: nn.Module) -> TrainState:
    init_rng, rng = jax.random.split(rng)
    variables = model.init({"params": init_rng}, x, train=False)
    params = variables.pop("params")
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
        rng=rng,
    )
    return state


init_pp_fn = shard_map(
    functools.partial(init_fn, model=model_pp),
    mesh,
    in_specs=(P(), P(config.data_axis_name)),
    out_specs=P(),
    check_rep=False,
)
state_pp_shapes = jax.eval_shape(init_pp_fn, model_init_rng, batch.inputs)
state_pp_specs = nn.get_partition_spec(state_pp_shapes)


init_pp_fn = jax.jit(
    shard_map(
        functools.partial(init_fn, model=model_pp),
        mesh,
        in_specs=(P(), P(config.data_axis_name)),
        out_specs=state_pp_specs,
        check_rep=False,
    ),
)
state_pp = init_pp_fn(model_init_rng, batch.inputs)

pprint(
    tree_map(lambda x: x.shape, state_pp.params["pipeline"]["sharded"]["mlp_layers"]["block"])
)



def loss_fn(
    params: PyTree, apply_fn: Any, batch: Batch, rng: jax.Array
) -> Tuple[jax.Array, Dict[str, Any]]:
    
    dropout_rng = fold_rng_over_axis(rng, (config.data_axis_name, config.model_axis_name))
    # Remaining computation is the same as before for single device.
    logits = apply_fn(
        {"params": params},
        batch.inputs,
        train=True,
        rngs={"dropout": dropout_rng},
    )
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch.labels)
    correct_pred = jnp.equal(jnp.argmax(logits, axis=-1), batch.labels)
    batch_size = batch.inputs.shape[0]
    # Mask out loss and accuracy for pipeline stages except last one.
    model_idx = jax.lax.axis_index(config.model_axis_name)
    model_size = jax.lax.psum(1, config.model_axis_name)
    loss = jnp.where(model_idx != model_size - 1, 0.0, loss)
    correct_pred = jnp.where(model_idx != model_size - 1, False, correct_pred)
    batch_size = jnp.where(model_idx != model_size - 1, 0, batch_size)
    # Collect metrics and return loss.
    step_metrics = {
        "loss": (loss.sum(), batch_size),
        "accuracy": (correct_pred.sum(), batch_size),
    }
    loss = loss.mean()
    return loss, step_metrics


def train_step_pp(
    state: TrainState,
    metrics: Metrics | None,
    batch: Batch,
) -> Tuple[TrainState, Metrics]:
    rng, step_rng = jax.random.split(state.rng)
    grads, step_metrics = accumulate_gradients(
        state,
        batch,
        step_rng,
        config.optimizer.num_minibatches,
        loss_fn=loss_fn,
    )
    # Update parameters. We need to sync the gradients across data devices before updating.
    with jax.named_scope("sync_gradients"):
        grads = sync_gradients(grads, (config.data_axis_name, config.model_axis_name))
    new_state = state.apply_gradients(grads=grads, rng=rng)
    # Sum metrics across replicas (both model and data axes).
    with jax.named_scope("sync_metrics"):
        step_metrics = tree_map(
            lambda x: jax.lax.psum(x, axis_name=(config.data_axis_name, config.model_axis_name)),
            step_metrics,
        )
    if metrics is None:
        metrics = step_metrics
    else:
        metrics = tree_map(jnp.add, metrics, step_metrics)
    return new_state, metrics

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
_, metric_shapes = jax.eval_shape(
    train_step_pp_fn,
    state_pp,
    None,
    batch,
)
metrics_pp = tree_map(lambda x: jnp.zeros(x.shape, dtype=x.dtype), metric_shapes)
state_pp, metrics_pp = train_step_pp_fn(state_pp, metrics_pp, batch)

for _ in range(15):
    state_pp, metrics_pp = train_step_pp_fn(state_pp, metrics_pp, batch)
final_metrics_pp = tree_map(lambda x: jnp.zeros(x.shape, dtype=x.dtype), metric_shapes)
state_pp, final_metrics_pp = train_step_pp_fn(state_pp, final_metrics_pp, batch)
print_metrics(final_metrics_pp, title="Final Metrics - Pipeline")
