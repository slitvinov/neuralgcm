from typing import Any, Mapping, TypeVar, Union
from dinosaur import scales
import jax.numpy as jnp
import numpy as np
import tree_math

Array = Union[np.ndarray, jnp.ndarray]
Quantity = scales.Quantity
PRNGKeyArray = Any
PyTreeState = TypeVar('PyTreeState')
Pytree = Any
State = TypeVar('State')
Params = Union[Mapping[str, Mapping[str, Array]], None]
