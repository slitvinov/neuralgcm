from typing import Any, TypeVar
from dinosaur import scales
import jax.numpy as jnp
import numpy as np
import tree_math

Array = Union[np.ndarray, jnp.ndarray]
State = TypeVar("State")
