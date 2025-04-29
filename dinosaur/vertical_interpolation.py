from __future__ import annotations
import dataclasses
import functools
import importlib
import dinosaur
from dinosaur import pytree_utils
from dinosaur import sigma_coordinates
import jax
import jax.numpy as jnp
import numpy as np


@dataclasses.dataclass(frozen=True)
class HybridCoordinates:
    a_boundaries: np.ndarray
    b_boundaries: np.ndarray

    def __post_init__(self):
        pass

    @classmethod
    def _from_resource_csv(cls, path: str) -> HybridCoordinates:
        levels_csv = importlib.resources.files(dinosaur).joinpath(path)
        with levels_csv.open() as f:
            a_in_pa, b = np.loadtxt(f,
                                    skiprows=1,
                                    usecols=(1, 2),
                                    delimiter="\t").T
        a = a_in_pa / 100
        assert 100 < a.max() < 1000
        return cls(a_boundaries=a, b_boundaries=b)

    @classmethod
    def ECMWF137(cls) -> HybridCoordinates:
        return cls._from_resource_csv("data/ecmwf137_hybrid_levels.csv")

    @property
    def layers(self) -> int:
        return len(self.a_boundaries) - 1

    def __hash__(self):
        return hash((tuple(self.a_boundaries.tolist()),
                     tuple(self.b_boundaries.tolist())))

    def get_sigma_boundaries(self, surface_pressure):
        return self.a_boundaries / surface_pressure + self.b_boundaries


def _interval_overlap(source_bounds, target_bounds) -> jnp.ndarray:
    upper = jnp.minimum(target_bounds[1:, jnp.newaxis],
                        source_bounds[jnp.newaxis, 1:])
    lower = jnp.maximum(target_bounds[:-1, jnp.newaxis],
                        source_bounds[jnp.newaxis, :-1])
    return jnp.maximum(upper - lower, 0)


def conservative_regrid_weights(source_bounds, target_bounds) -> jnp.ndarray:
    weights = _interval_overlap(source_bounds, target_bounds)
    weights /= jnp.sum(weights, axis=1, keepdims=True)
    assert weights.shape == (target_bounds.size - 1, source_bounds.size - 1)
    return weights


@functools.partial(jax.jit, static_argnums=(1, 2))
def regrid_hybrid_to_sigma(
    fields,
    hybrid_coords,
    sigma_coords,
    surface_pressure,
):

    @jax.jit
    @functools.partial(jnp.vectorize, signature="(x,y),(a),(b,x,y)->(c,x,y)")
    @functools.partial(jax.vmap, in_axes=(-1, None, -1), out_axes=-1)
    @functools.partial(jax.vmap, in_axes=(-1, None, -1), out_axes=-1)
    def regrid(surface_pressure, sigma_bounds, field):
        assert sigma_bounds.shape == (sigma_coords.layers + 1, )
        hybrid_bounds = hybrid_coords.get_sigma_boundaries(surface_pressure)
        weights = conservative_regrid_weights(hybrid_bounds, sigma_bounds)
        result = jnp.einsum("ab,b->a", weights, field, precision="float32")
        assert result.shape[0] == sigma_coords.layers
        return result

    return pytree_utils.tree_map_over_nonscalars(
        lambda x: regrid(surface_pressure, sigma_coords.boundaries, x), fields)
