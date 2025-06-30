import jax
import jax.numpy as jnp
from flax.struct import dataclass
from flax.training import train_state
from typing import Any , Callable , Dict , Tuple

import os

Pytree = Any
Metrics = Dict[str , Tuple[jax.Array , ...]]

class TrainState(train_state.TrainState):
    rng : jax.Array


@dataclass
class Batch:
    inputs : jax.Array
    labels : jax.Array


def sim_multiCPU_dev(
        device_count : int = 8
):    
    flags = os.environ.get("XLA_FLAGS" , "")
    flags += f" --xla_force_host_platform_device_count={device_count}"
    
    os.environ["XLA_FLAGS"] = flags
    
    #disable CUDA 
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def accum_grads_loop(
        batch : Batch,
        state : TrainState,
        key,
        n_minbatch : int,
        loss_fn : Callable,

) -> Tuple[Pytree , Metrics]:

    bs = batch.inputs.shape[0]
    min_batchSize = bs // n_minbatch
    keys = jax.random.split(key , n_minbatch)

    grad_fn = jax.value_and_grad(loss_fn , has_aux=True)

    grads = None
    metrics : Metrics = None

    for batch_idx in range(n_minbatch):

        start_idx = batch_idx * min_batchSize
        end_idx = start_idx + min_batchSize
        #slicing Batch class 
        miniBatch = jax.tree_util.tree_map(lambda x : x[start_idx: end_idx] , batch)

        (_ , step_metric) , step_grad = grad_fn(
            state.params , state.apply_fn , miniBatch , keys[batch_idx]
        )

        if grads is None:
            grads = step_grad
            metrics = step_metric

        else:
            grads = jax.tree_util.tree_map(jnp.add , grads , step_grad)
            metrics = jax.tree_util.tree_map(jnp.add , metrics , step_metric)
            
    grads = jax.tree_util.tree_map(lambda g : g / n_minbatch , grads)
    return grads , metrics
    




