from functools import partial
from typing import Any, Callable, Dict, Tuple


import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict

from data_paral import shard_module_params
from pipeline_parallel import ModelParallelWrapper
from tensor_paral import MLPBlockInput , MLPBlockOutput
from tensor_parallel_async import TPAsyncDense ,TPAsyncMLPBlock , TPNorm

PyTree = Any
Parameter = jax.Array | nn.Partitioned
Metrics = Dict[str, Tuple[jax.Array, ...]]


class QKVDense(nn.Module):
    config: ConfigDict
    num_heads: int
    head_dim: int
    kernel_init: Callable
    use_bias: bool = False

    @nn.compact
    def __call__(self, x: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:

        q = nn.DenseGeneral(
            (self.num_heads, self.head_dim),
            kernel_init=self.kernel_init,
            use_bias=False,
            dtype=self.config.dtype,
            name="query",
        )(x)

        k = nn.DenseGeneral(
            (self.num_heads, self.head_dim),
            kernel_init=self.kernel_init,
            use_bias=False,
            dtype=self.config.dtype,
            name="key",
        )(x)

        v = nn.DenseGeneral(
            (self.num_heads, self.head_dim),
            kernel_init=self.kernel_init,
            use_bias=False,
            dtype=self.config.dtype,
            name="value",
        )(x)

        if self.config.normalize_qk:
            q = nn.RMSNorm(
                dtype=self.config.dtype,
                name="query_norm",
            )(q)

            k = nn.RMSNorm(
                dtype=self.config.dtype,
                name="key_norm",
            )(k)

        return q, k, v

def dot_product_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    mask: jax.Array | None,
    softmax_dtype: jnp.dtype = jnp.float32,
):

    num_features = query.shape[-1]

    dtype = query.dtype
    scale = num_features**-0.5

    query = query * scale
    query = query.astype(softmax_dtype)

    key = key.astype(softmax_dtype)
    weights = jnp.einsum("...qhd,...khd->...hqk", query, key)

    if mask is not None:
        weights = jnp.where(mask, weights, jnp.finfo(softmax_dtype).min)

    weights = nn.softmax(weights, axis=-1)
    weights = weights.astype(dtype)

    new_vals = jnp.einsum("...hqk,...khd->...qhd", weights, value)
    new_vals = new_vals.astype(dtype)

    return new_vals


class AttnOut(nn.Module):

    config: ConfigDict
    features: int
    kernel_init: Callable = nn.initializers.lecun_normal()
    use_bias: bool = True

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        x = nn.DenseGeneral(
            features=self.features,
            axis=(-2, -1),
            kernel_init=self.kernel_init,
            use_bias=self.use_bias,
            dtype=self.config.dtype,
            name="out",
        )(x)
        return x


class TPMultiHeadAttn(nn.Module):
    config: ConfigDict
    train: bool
    mask: jax.Array | None = None

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:

        tp_size = jax.lax.psum(1, self.config.model_axis_name)
        input_features = x.shape[-1]
        head_dim = self.config.head_dim
        num_heads = self.config.num_heads
        
        x = TPNorm(config=self.config, name="pre_norm")(x)
        
        q, k, v = TPAsyncDense(
            dense_fn= partial(
                QKVDense,
                config=self.config,
                num_heads=num_heads // tp_size,
                head_dim=head_dim,
            ),
            model_axis_name=self.config.model_axis_name,
            tp_mode="gather",
            kernel_init_adjustment=tp_size**-0.5,
            name="qkv",
        )(x)
        
        x = dot_product_attention(q, k, v, self.mask)
        
        x = TPAsyncDense(
            dense_fn= partial(
                AttnOut,
                config=self.config,
                features=input_features,
            ),
            model_axis_name=self.config.model_axis_name,
            tp_mode="scatter",
            kernel_init_adjustment=tp_size**-0.5,
            name="out",
        )(x)
        return x


def prepare_module(
    layer: Callable[..., nn.Module], layer_name: str, config: ConfigDict
) -> Callable[..., nn.Module]:
    if config.get("fsdp", None) is not None and layer_name in config.fsdp.modules:
        layer = shard_module_params(
            layer,
            axis_name=config.data_axis_name,
            min_weight_size=config.fsdp.min_weight_size,
        )

    if config.get("remat", None) is not None and layer_name in config.remat:
        layer = nn.remat(layer, prevent_cse=False)

    return layer


class TPTransformerBlock(nn.Module):
    config: ConfigDict
    train: bool
    mask: jax.Array | None = None

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:

        attn_layer = prepare_module(TPMultiHeadAttn, "Attn", self.config)
        attn_out = attn_layer(
            config=self.config,
            train=self.train,
            mask=self.mask,
            name="attn",
        )(x)

        attn_out = nn.Dropout(
            rate=self.config.dropout_rate, deterministic=not self.train
        )(attn_out)

        x = x + attn_out
        mlp_layer = prepare_module(TPAsyncMLPBlock, "MLP", self.config)
        mlp_out = mlp_layer(
            config=self.config,
            train=self.train,
            name="mlp",
        )(x)

        mlp_out = nn.Dropout(
            rate=self.config.dropout_rate, deterministic=not self.train
        )(mlp_out)

        x = x + mlp_out

        return x


class QKVMLPDense(nn.Module):

    config: ConfigDict
    num_heads: int
    head_dim: int
    mlp_dim: int
    kernel_init: Callable
    use_bias: bool = False

    @nn.compact
    def __call__(
        self, x: jax.Array
    ) -> Tuple[jax.Array, Tuple[jax.Array, jax.Array, jax.Array]]:

        h = MLPBlockInput(
            config=self.config,
            features=self.mlp_dim,
            kernel_init=self.kernel_init,
            use_bias=self.use_bias,
            use_norm=False,
            name="mlp",
        )(x)

        q, k, v = QKVDense(
            config=self.config,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            kernel_init=self.kernel_init,
            use_bias=self.use_bias,
            name="qkv",
        )(x)

        return h, (q, k, v)
