from __future__ import annotations
import dataclasses
import functools
from typing import Callable
from dinosaur import jax_numpy_utils
from dinosaur import typing
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np

Array = typing.Array
einsum = functools.partial(jnp.einsum, precision=lax.Precision.HIGHEST)


def _slice_shape_along_axis(
    x: np.ndarray,
    axis: int,
    slice_width: int = 1,
) -> tuple[int, ...]:
    x_shape = list(x.shape)
    x_shape[axis] = slice_width
    return tuple(x_shape)


def _with_f64_math(
    f: Callable[[np.ndarray],
                np.ndarray], ) -> Callable[[np.ndarray], np.ndarray]:
    return lambda x: f(x.astype(np.float64)).astype(x.dtype)


@dataclasses.dataclass(frozen=True)
class SigmaCoordinates:
    boundaries: np.ndarray

    def __init__(self, boundaries):
        boundaries = np.asarray(boundaries)
        object.__setattr__(self, 'boundaries', boundaries)

    @property
    def centers(self) -> np.ndarray:
        return _with_f64_math(lambda x: (x[1:] + x[:-1]) / 2)(self.boundaries)

    @property
    def layer_thickness(self) -> np.ndarray:
        return _with_f64_math(np.diff)(self.boundaries)

    @property
    def center_to_center(self) -> np.ndarray:
        return _with_f64_math(np.diff)(self.centers)

    @property
    def layers(self) -> int:
        return len(self.boundaries) - 1

    @classmethod
    def equidistant(cls, layers: int, dtype=np.float32) -> SigmaCoordinates:
        boundaries = np.linspace(0, 1, layers + 1, dtype=dtype)
        return cls(boundaries)

    def asdict(self):
        return {k: v.tolist() for k, v in dataclasses.asdict(self).items()}

    def __hash__(self):
        return hash(tuple(self.centers.tolist()))


@jax.named_call
def centered_difference(x: np.ndarray,
                        coordinates: SigmaCoordinates,
                        axis: int = -3) -> np.ndarray:
    dx = jax_numpy_utils.diff(x, axis=axis)
    dx_axes = range(dx.ndim)
    inv_d𝜎 = 1 / coordinates.center_to_center
    inv_d𝜎_axes = [dx_axes[axis]]
    return einsum(dx,
                  dx_axes,
                  inv_d𝜎,
                  inv_d𝜎_axes,
                  dx_axes,
                  precision='float32')


@jax.named_call
def cumulative_sigma_integral(
    x: Array,
    coordinates: SigmaCoordinates,
    axis: int = -3,
    downward: bool = True,
    cumsum_method: str = 'dot',
    sharding: jax.sharding.NamedSharding | None = None,
) -> jax.Array:
    x_axes = range(x.ndim)
    d𝜎 = coordinates.layer_thickness
    d𝜎_axes = [x_axes[axis]]
    xd𝜎 = einsum(x, x_axes, d𝜎, d𝜎_axes, x_axes)
    if downward:
        return jax_numpy_utils.cumsum(xd𝜎,
                                      axis,
                                      method=cumsum_method,
                                      sharding=sharding)
    else:
        assert False


@jax.named_call
def sigma_integral(
    x: Array,
    coordinates: SigmaCoordinates,
    axis: int = -3,
    keepdims: bool = True,
) -> jax.Array:
    x_axes = range(x.ndim)
    d𝜎 = coordinates.layer_thickness
    d𝜎_axes = [x_axes[axis]]
    xd𝜎 = einsum(x, x_axes, d𝜎, d𝜎_axes, x_axes)
    return xd𝜎.sum(axis=axis, keepdims=keepdims)


@jax.named_call
def centered_vertical_advection(
    w: Array,
    x: Array,
    coordinates: SigmaCoordinates,
    axis: int = -3,
    w_boundary_values: tuple[Array, Array] | None = None,
    dx_dsigma_boundary_values: tuple[Array, Array] | None = None,
) -> jnp.ndarray:
    if w_boundary_values is None:
        w_slc_shape = _slice_shape_along_axis(w, axis)
        w_boundary_values = (
            jnp.zeros(w_slc_shape,
                      dtype=jax.dtypes.canonicalize_dtype(w.dtype)),
            jnp.zeros(w_slc_shape,
                      dtype=jax.dtypes.canonicalize_dtype(w.dtype)),
        )
    if dx_dsigma_boundary_values is None:
        x_slc_shape = _slice_shape_along_axis(x, axis)
        dx_dsigma_boundary_values = (
            jnp.zeros(x_slc_shape,
                      dtype=jax.dtypes.canonicalize_dtype(x.dtype)),
            jnp.zeros(x_slc_shape,
                      dtype=jax.dtypes.canonicalize_dtype(x.dtype)),
        )
    w_boundary_top, w_boundary_bot = w_boundary_values
    w = jnp.concatenate([w_boundary_top, w, w_boundary_bot], axis=axis)
    x_diff = centered_difference(x, coordinates, axis)
    x_diff_boundary_top, x_diff_boundary_bot = dx_dsigma_boundary_values
    x_diff = jnp.concatenate(
        [x_diff_boundary_top, x_diff, x_diff_boundary_bot], axis=axis)
    w_times_x_diff = w * x_diff
    return -0.5 * (lax.slice_in_dim(w_times_x_diff, 1, None, axis=axis) +
                   lax.slice_in_dim(w_times_x_diff, 0, -1, axis=axis))
