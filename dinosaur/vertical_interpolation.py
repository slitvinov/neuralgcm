from __future__ import annotations
import dataclasses
import functools
import importlib
from typing import Any, Callable, Dict, Sequence, TypeVar, Union
import dinosaur
from dinosaur import pytree_utils
from dinosaur import sigma_coordinates
from dinosaur import typing
import jax
import jax.numpy as jnp
import numpy as np

Array = typing.Array
InterpolateFn = Callable[[Array, Array, Array], Array]

@dataclasses.dataclass(frozen=True)
class PressureCoordinates:
    centers: np.ndarray

    def __init__(self, centers: Union[Sequence[float], np.ndarray]):
        object.__setattr__(self, 'centers', np.asarray(centers))
        if not all(np.diff(self.centers) > 0):
            raise ValueError(
                'Expected `centers` to be monotonically increasing, '
                f'got centers = {self.centers}')

    @property
    def layers(self) -> int:
        return len(self.centers)

    def asdict(self) -> Dict[str, Any]:
        return {k: v.tolist() for k, v in dataclasses.asdict(self).items()}

    def __hash__(self):
        return hash(tuple(self.centers.tolist()))

    def __eq__(self, other):
        return isinstance(other, PressureCoordinates) and np.array_equal(
            self.centers, other.centers)


@dataclasses.dataclass(frozen=True)
class HybridCoordinates:
    a_boundaries: np.ndarray
    b_boundaries: np.ndarray

    def __post_init__(self):
        if len(self.a_boundaries) != len(self.b_boundaries):
            raise ValueError(
                'Expected `a_boundaries` and `b_boundaries` to have the same length, '
                f'got {len(self.a_boundaries)} and {len(self.b_boundaries)}.')

    @classmethod
    def _from_resource_csv(cls, path: str) -> HybridCoordinates:
        levels_csv = importlib.resources.files(dinosaur).joinpath(path)
        with levels_csv.open() as f:
            a_in_pa, b = np.loadtxt(f,
                                    skiprows=1,
                                    usecols=(1, 2),
                                    delimiter='\t').T
        a = a_in_pa / 100  # convert from Pa to hPa
        assert 100 < a.max() < 1000
        return cls(a_boundaries=a, b_boundaries=b)

    @classmethod
    def ECMWF137(cls) -> HybridCoordinates:  # pylint: disable=invalid-name
        return cls._from_resource_csv('data/ecmwf137_hybrid_levels.csv')

    @classmethod
    def UFS127(cls) -> HybridCoordinates:  # pylint: disable=invalid-name
        return cls._from_resource_csv('data/ufs127_hybrid_levels.csv')

    @property
    def layers(self) -> int:
        return len(self.a_boundaries) - 1

    def __hash__(self):
        return hash((tuple(self.a_boundaries.tolist()),
                     tuple(self.b_boundaries.tolist())))

    def __eq__(self, other):
        return (isinstance(other, HybridCoordinates)
                and np.array_equal(self.a_boundaries, other.a_boundaries)
                and np.array_equal(self.b_boundaries, other.b_boundaries))

    def get_sigma_boundaries(self,
                             surface_pressure: typing.Numeric) -> typing.Array:
        return self.a_boundaries / surface_pressure + self.b_boundaries

SigmaOrPressure = TypeVar(
    'SigmaOrPressure',
    sigma_coordinates.SigmaCoordinates,
    PressureCoordinates,
)

def _interval_overlap(source_bounds: typing.Array,
                      target_bounds: typing.Array) -> jnp.ndarray:
    upper = jnp.minimum(target_bounds[1:, jnp.newaxis],
                        source_bounds[jnp.newaxis, 1:])
    lower = jnp.maximum(target_bounds[:-1, jnp.newaxis],
                        source_bounds[jnp.newaxis, :-1])
    return jnp.maximum(upper - lower, 0)


def conservative_regrid_weights(source_bounds: typing.Array,
                                target_bounds: typing.Array) -> jnp.ndarray:
    weights = _interval_overlap(source_bounds, target_bounds)
    weights /= jnp.sum(weights, axis=1, keepdims=True)
    assert weights.shape == (target_bounds.size - 1, source_bounds.size - 1)
    return weights


@functools.partial(jax.jit, static_argnums=(1, 2))
def regrid_hybrid_to_sigma(
    fields: typing.Pytree,
    hybrid_coords: HybridCoordinates,
    sigma_coords: sigma_coordinates.SigmaCoordinates,
    surface_pressure: typing.Array,
) -> typing.Pytree:

    @jax.jit
    @functools.partial(jnp.vectorize, signature='(x,y),(a),(b,x,y)->(c,x,y)')
    @functools.partial(jax.vmap, in_axes=(-1, None, -1), out_axes=-1)
    @functools.partial(jax.vmap, in_axes=(-1, None, -1), out_axes=-1)
    def regrid(surface_pressure, sigma_bounds, field):
        assert sigma_bounds.shape == (sigma_coords.layers + 1, )
        if field.shape[0] != hybrid_coords.layers:
            raise ValueError(
                f'Source has {hybrid_coords.layers} layers, but field has'
                f' {fields.shape[0]}')
        hybrid_bounds = hybrid_coords.get_sigma_boundaries(surface_pressure)
        weights = conservative_regrid_weights(hybrid_bounds, sigma_bounds)
        result = jnp.einsum('ab,b->a', weights, field, precision='float32')
        assert result.shape[0] == sigma_coords.layers
        return result

    return pytree_utils.tree_map_over_nonscalars(
        lambda x: regrid(surface_pressure, sigma_coords.boundaries, x), fields)
