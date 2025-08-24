from util import sim_multiCPU_dev , Batch , TrainState , accum_grads , print_metrics
sim_multiCPU_dev()

from functools import partial
from pprint import pprint
from typing import Any ,Callable , Dict , Literal , Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh
from ml_collections import ConfigDict


from data_paral import fold_rng_over_axis
from pipeline_parallel import ModelParallelWrapper

PyTree = Any
Parameter = jax.Array | nn.Partitioned
Metrics = Dict[str , Tuple[jax.Array , ...]]


def scale_init(
    init_fn : Callable,
    scale_factor : float = 1.0
):
    
    def _init_fn(rng , *args , **kwargs):
        return scale_factor * init_fn(rng , *args , **kwargs)
    
    return _init_fn

class TPDense(nn.Module):

    dense_fn : Any
    model_axis_name  : str
    tp_mode: Literal["scatter", "gather", "none"] = "none"
    skip_communication: bool = False
    kernel_init: Callable = nn.initializers.lecun_normal()
    kernel_init_adjustment: float = 1.0
    dense_name: str = "module"

    @nn.compact
    def __call__(self , x : jax.Array) -> jax.Array:
        tp_size = jax.lax.psum(1,self.model_axis_name)
        tp_mode = self.tp_mode if tp_size > 1 else "none"

        dense_fn = partial(
            ModelParallelWrapper,
            model_axis_name=self.model_axis_name,
            module_fn = partial(
                self.dense_fn,
                kernel_init = scale_init(self.kernel_init , self.kernel_init_adjustment),
            ),
            name=self.dense_name
        )

        if tp_mode == "none":
            x = self.dense_fn(kernel_init=self.kernel_init)(x)

        elif tp_mode == "gather":
            if not self.skip_communication:
                x = jax.lax.all_gather(x , self.model_axis_name , axis=-1 , tiled=True)
            x = dense_fn()(x)

        elif tp_mode == "scatter":

            x = dense_fn()(x)

            if not self.skip_communication:
                x = jax.lax.psum_scatter(
                    x,
                    axis_name = self.model_axis_name,
                    scatter_dimension=x.ndim -1,
                    tiled=True
                )

        else:
            raise ValueError(f"Unknown Tensor Parallel model : {tp_mode}")

        return x


class MLPBlockInput(nn.Module):
    config: ConfigDict
    features: int
    kernel_init: Callable = nn.initializers.lecun_normal()
    use_bias: bool = True
    use_norm: bool = True

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        if self.use_norm:
            x = nn.RMSNorm(dtype=self.config.dtype, name="pre_norm")(x)
        x = nn.Dense(
            features=self.features,
            kernel_init=self.kernel_init,
            use_bias=self.use_bias,
            dtype=self.config.dtype,
            name="dense",
        )(x)
        return x


class MLPBlockOutput(nn.Module):
    config: ConfigDict
    features: int
    kernel_init: Callable = nn.initializers.lecun_normal()
    use_bias: bool = True

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        x = nn.silu(x)
        x = nn.Dense(
            features=self.features,
            kernel_init=self.kernel_init,
            use_bias=self.use_bias,
            dtype=self.config.dtype,
            name="dense",
        )(x)
        return x


class TPMLPBlock(nn.Module):
    config: ConfigDict
    train: bool

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        tp_size = jax.lax.psum(1, self.config.model_axis_name)
        input_features = x.shape[-1]
        # Input layer
        x = TPDense(
            dense_fn=functools.partial(
                MLPBlockInput,
                config=self.config,
                features=self.config.hidden_size * self.config.mlp_expansion // tp_size,
            ),
            model_axis_name=self.config.model_axis_name,
            tp_mode="gather",
            name="input",
        )(x)
        # Output layer
        x = TPDense(
            dense_fn=functools.partial(
                MLPBlockOutput,
                config=self.config,
                features=input_features * tp_size,
            ),
            model_axis_name=self.config.model_axis_name,
            tp_mode="scatter",
            kernel_init_adjustment=tp_size**-0.5,  # fan-in with tp_size fewer inputs.
            name="output",
        )(x)
        return x


class TPMLPLayers(nn.Module):
    config: ConfigDict
    train: bool
    block_class: Callable[..., nn.Module] = TPMLPBlock

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        module = self.block_class(config=self.config, train=self.train, name="block")
        x, _ = nn.scan(
            lambda module, carry, _: (module(carry) + carry, None),
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            length=self.config.num_layers,
            metadata_params={
                "partition_name": None
            },  # We do not need to partition the parameters over the layer axis.
        )(module, x, ())
        return x


class TPClassifier(nn.Module):
    config: ConfigDict
    block_class: Callable[..., nn.Module] = TPMLPBlock

    @nn.compact
    def __call__(self, x: jax.Array, train: bool) -> jax.Array:
        tp_size = jax.lax.psum(1, self.config.model_axis_name)
        # Input layer
        x = TPDense(
            dense_fn=functools.partial(
                nn.Dense,
                features=self.config.hidden_size // tp_size,
                dtype=self.config.dtype,
            ),
            model_axis_name=self.config.model_axis_name,
            tp_mode="gather",
            skip_communication=True,  # Input already gathered.
            name="input_layer",
        )(x)
        # Backbone MLP blocks
        x = TPMLPLayers(
            config=self.config, train=train, name="mlp", block_class=self.block_class
        )(x)
        # Output layer
        x = TPDense(
            dense_fn=functools.partial(
                nn.Dense,
                features=self.config.num_classes,
                dtype=self.config.dtype,
            ),
            model_axis_name=self.config.model_axis_name,
            tp_mode="scatter",
            skip_communication=True,  # Manual communication.
            name="output_layer",
            kernel_init_adjustment=tp_size**-0.5,  # fan-in with tp_size fewer inputs.
        )(x)
        x = jax.lax.psum(x, axis_name=self.config.model_axis_name)
        x = x.astype(jnp.float32)
        return x


