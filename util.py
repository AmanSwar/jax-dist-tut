import jax
from flax.struct import dataclass
from flax.training import train_state
from typing import Any , Callable , Dict , Tuple

Metrics = Dict[str , Tuple[jax.Array , ...]]

class TrainState(train_state.TrainState):
    rng : jax.Array


@dataclass
class Batch:
    inputs : jax.Array
    labels : jax.Array

