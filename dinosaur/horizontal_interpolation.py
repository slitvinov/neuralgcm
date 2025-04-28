import dataclasses
import functools
from dinosaur import spherical_harmonic
from dinosaur import typing
import jax
import jax.numpy as jnp
import numpy as np
from sklearn import neighbors

@dataclasses.dataclass(frozen=True)
class Regridder:
    source_grid: spherical_harmonic.Grid
    target_grid: spherical_harmonic.Grid

    def __call__(self, field: typing.Array) -> jnp.ndarray:
        raise NotImplementedError


