from __future__ import annotations
import dataclasses
import functools
from typing import Callable
from dinosaur import jax_numpy_utils
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np

einsum = functools.partial(jnp.einsum, precision=lax.Precision.HIGHEST)


def _slice_shape_along_axis(
    x: np.ndarray,
    axis: int,
    slice_width: int = 1,
):
    x_shape = list(x.shape)
    x_shape[axis] = slice_width
    return tuple(x_shape)


def _with_f64_math(f: Callable[[np.ndarray], np.ndarray], ):
    return lambda x: f(x.astype(np.float64)).astype(x.dtype)


@dataclasses.dataclass(frozen=True)
class SigmaCoordinates:
    boundaries: np.ndarray

    def __init__(self, boundaries):
        boundaries = np.asarray(boundaries)
        object.__setattr__(self, "boundaries", boundaries)

    @property
    def centers(self):
        return _with_f64_math(lambda x: (x[1:] + x[:-1]) / 2)(self.boundaries)

    @property
    def layer_thickness(self):
        return _with_f64_math(np.diff)(self.boundaries)

    @property
    def center_to_center(self):
        return _with_f64_math(np.diff)(self.centers)

    @property
    def layers(self):
        return len(self.boundaries) - 1

    @classmethod
    def equidistant(cls, layers: int, dtype=np.float32):
        boundaries = np.linspace(0, 1, layers + 1, dtype=dtype)
        return cls(boundaries)

    def asdict(self):
        return {k: v.tolist() for k, v in dataclasses.asdict(self).items()}

    def __hash__(self):
        return hash(tuple(self.centers.tolist()))


def centered_difference(x: np.ndarray,
                        coordinates: SigmaCoordinates,
                        axis: int = -3):
    dx = jax_numpy_utils.diff(x, axis=axis)
    dx_axes = range(dx.ndim)
    inv_d𝜎 = 1 / coordinates.center_to_center
    inv_d𝜎_axes = [dx_axes[axis]]
    return einsum(dx,
                  dx_axes,
                  inv_d𝜎,
                  inv_d𝜎_axes,
                  dx_axes,
                  precision="float32")


def cumulative_sigma_integral(
    x,
    coordinates,
    axis= -3,
    downward= True,
    cumsum_method = "dot",
    sharding,
):
    x_axes = range(x.ndim)
    d𝜎 = coordinates.layer_thickness
    d𝜎_axes = [x_axes[axis]]
    xd𝜎 = einsum(x, x_axes, d𝜎, d𝜎_axes, x_axes)
    return jax_numpy_utils.cumsum(xd𝜎,
                                  axis,
                                  method=cumsum_method,
                                  sharding=sharding)


def sigma_integral(
    x,
    coordinates: SigmaCoordinates,
    axis: int = -3,
    keepdims: bool = True,
):
    x_axes = range(x.ndim)
    d𝜎 = coordinates.layer_thickness
    d𝜎_axes = [x_axes[axis]]
    xd𝜎 = einsum(x, x_axes, d𝜎, d𝜎_axes, x_axes)
    return xd𝜎.sum(axis=axis, keepdims=keepdims)


def centered_vertical_advection(
    w,
    x,
    coordinates: SigmaCoordinates,
    axis: int = -3,
    w_boundary_values=None,
    dx_dsigma_boundary_values=None,
):
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
