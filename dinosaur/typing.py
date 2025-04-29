import dataclasses
from typing import Any, Callable, Generic, Mapping, TypeVar, Union
from dinosaur import scales
import jax.numpy as jnp
import numpy as np
import tree_math

Array = Union[np.ndarray, jnp.ndarray]
ArrayOrArrayTuple = Union[Array, tuple[Array, ...]]
Numeric = Union[float, int, Array]
Quantity = scales.Quantity
PRNGKeyArray = Any
PyTreeState = TypeVar('PyTreeState')
Pytree = Any
PyTreeMemory = Pytree
PyTreeDiagnostics = Pytree
AuxFeatures = dict[str, Any]
DataState = dict[str, Any]
ForcingData = dict[str, Any]
State = TypeVar('State')
Forcing = Pytree
Params = Union[Mapping[str, Mapping[str, Array]], None]
