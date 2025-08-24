from functools import partial
from typing import Any , Callable , Dict , List,  Literal , Sequence , Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict

from pipeline_parallel import ModelParallelWrapper
from tensor_paral import MLPBlockInput , MLPBlockOutput , scale_init


PyTree = Any
Parameter = jax.Array | nn.Partitioned
Metrics = Dict[str, Tuple[jax.Array, ...]]


def async_gather(
    x  : PyTree,
    axis_name : str,
    shift_up : bool = True
) -> List[PyTree]:

    tp_size = jax.lax.psum(1, axis_name)
    if shift_up:
        shift_perm = [(j, (j + 1) % tp_size) for j in range(tp_size)]
    else:
        shift_perm = [(j, (j - 1) % tp_size) for j in range(tp_size)]
    ps = [x]
    p = x
    for _ in range(1, tp_size):
            p = jax.lax.ppermute(p, axis_name, perm=shift_perm)
            ps.append(p)
    return ps

def async_gather_bidirectional(
    x: jax.Array, axis_name: str, shift_up: bool = True
) -> List[jax.Array]:

    tp_size = jax.lax.psum(1, axis_name)
    shift_up_perm = [(j, (j + 1) % tp_size) for j in range(tp_size)]
    shift_down_perm = [(j, (j - 1) % tp_size) for j in range(tp_size)]
    ps_up = []
    ps_down = []
    p_up = x
    p_down = x

    for i in range(1, tp_size):
        if i % 2 == 0:
            p_down = jax.lax.ppermute(p_down, axis_name=axis_name, perm=shift_down_perm)
            ps_down.append(p_down)
        else:
            p_up = jax.lax.ppermute(p_up, axis_name=axis_name, perm=shift_up_perm)
            ps_up.append(p_up)

    if shift_up:
        ps = [x] + ps_up + ps_down[::-1]
    else:
        ps = [x] + ps_down + ps_up[::-1]
    return ps


def async_gather_split(x: jax.Array, axis_name: str) -> List[jax.Array]:

    x1, x2 = jax.tree_util.tree_map(lambda x: jnp.split(x, 2, axis=-1), x)
    return async_gather(x1, axis_name, shift_up=True) + async_gather(
        x2, axis_name, shift_up=False
    )


def async_scatter(
    xs: Sequence[PyTree], axis_name: str, shift_up: bool = True
) -> PyTree:

    tp_size = jax.lax.psum(1, axis_name)

    assert (
        len(xs) == tp_size
    ), f"Number of shards needs to match axis size, but got {len(xs)} with {axis_name} axis size {tp_size}."

    if shift_up:
        shift_perm = [(j, (j + 1) % tp_size) for j in range(tp_size)]

    else:
        shift_perm = [(j, (j - 1) % tp_size) for j in range(tp_size)]

    y = xs[0]

    for x in xs[1:]:
        y = jax.lax.ppermute(y, axis_name, perm=shift_perm)
        y = jax.tree_util.tree_map(jnp.add, y, x)

    return y


def async_scatter_split(xs: Sequence[PyTree], axis_name: str) -> PyTree:
    

    def _split(x: PyTree) -> Tuple[PyTree, PyTree]:

        return (
            jax.tree_util.tree_map(lambda x: x[..., : x.shape[-1] // 2], x),
            jax.tree_util.tree_map(lambda x: x[..., x.shape[-1] // 2 :], x),
        )

    tp_size = jax.lax.psum(1, axis_name)
    
    assert (
        len(xs) == tp_size
    ), f"Number of shards needs to match axis size, but got {len(xs)} with {axis_name} axis size {tp_size}."
    
    shift_perm_up = [(j, (j + 1) % tp_size) for j in range(tp_size)]
    shift_perm_down = [(j, (j - 1) % tp_size) for j in range(tp_size)]
    
    y_up, y_down = _split(xs[0])
    
    for x in xs[1:]:
        
        y_up = jax.lax.ppermute(y_up, axis_name, perm=shift_perm_up)
        
        y_down = jax.lax.ppermute(y_down, axis_name, perm=shift_perm_down)
        
        x_up, x_down = _split(x)
        y_up = jax.tree_util.tree_map(jnp.add, y_up, x_up)
        y_down = jax.tree_util.tree_map(jnp.add, y_down, x_down)
    
    return jax.tree_util.tree_map(lambda y1, y2: jnp.concatenate([y1, y2], axis=-1), y_up, y_down)
