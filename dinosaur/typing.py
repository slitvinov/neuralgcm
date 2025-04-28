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

@tree_math.struct
class RandomnessState:
    core: Union[Pytree, None] = None
    nodal_value: Union[Pytree, None] = None
    modal_value: Union[Pytree, None] = None
    prng_key: Union[PRNGKeyArray, None] = None
    prng_step: Union[int, None] = None


State = TypeVar('State')
StateFn = Callable[[State], State]
InverseFn = Callable[[State, jnp.ndarray], State]
StepFn = Callable[[State, State], State]
FilterFn = Callable[[State, State, State], tuple[State, State]]
ScanFn = Callable[..., Any]
PytreeFn = Callable[[Pytree], Pytree]
PyTreeTermsFn = Callable[[PyTreeState], PyTreeState]
PyTreeInverseFn = Callable[[PyTreeState, Numeric], PyTreeState]
TimeStepFn = Callable[[PyTreeState], PyTreeState]
PyTreeFilterFn = Callable[[PyTreeState], PyTreeState]
PyTreeStepFilterFn = Callable[[PyTreeState, PyTreeState], PyTreeState]
PyTreeStepFilterModule = Callable[..., PyTreeStepFilterFn]
Forcing = Pytree
ForcingFn = Callable[[ForcingData, float], Forcing]
ForcingModule = Callable[..., ForcingFn]
PostProcessFn = Callable[..., Any]
Params = Union[Mapping[str, Mapping[str, Array]], None]
StepFn = Callable[[PyTreeState, Union[Forcing, None]], PyTreeState]
StepModule = Callable[..., StepFn]
CorrectorFn = Callable[
    [PyTreeState, Union[PyTreeState, None], Union[Forcing, None]], PyTreeState]
CorrectorModule = Callable[..., CorrectorFn]
ParameterizationFn = Callable[
    [
        PyTreeState,
        Union[PyTreeMemory, None],
        Union[PyTreeDiagnostics, None],
        Union[RandomnessState, None],
        Union[Forcing, None],
    ],
    PyTreeState,
]
ParameterizationModule = Callable[..., ParameterizationFn]
TrajectoryFn = Callable[..., tuple[Any, Any]]
TransformFn = Callable[[Pytree], Pytree]
TransformModule = Callable[..., TransformFn]
GatingFactory = Callable[..., Callable[[Array, Array], Array]]
TowerFactory = Callable[..., Callable[..., Any]]
LayerFactory = Callable[..., Callable[..., Any]]
EmbeddingFn = Callable[
    [
        Pytree,
        Union[PyTreeMemory, None],
        Union[PyTreeDiagnostics, None],
        Union[RandomnessState, None],
        Union[Forcing, None],
    ],
    Pytree,
]
EmbeddingModule = Callable[..., EmbeddingFn]
