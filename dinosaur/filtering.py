import functools
from typing import Callable, Union
from dinosaur import spherical_harmonic
from dinosaur import typing
import jax
import jax.numpy as jnp
import numpy as np


def _preserves_shape(target, scaling):
    target_shape = np.shape(target)
    return target_shape == np.broadcast_shapes(target_shape, scaling.shape)


def _make_filter_fn(scaling, name=None):
    rescale = lambda x: scaling * x if _preserves_shape(x, scaling) else x
    return functools.partial(jax.tree_util.tree_map,
                             jax.named_call(rescale, name=name))


def exponential_filter(
    grid: spherical_harmonic.Grid,
    attenuation: Union[float, typing.Array] = 16,
    order: Union[int, typing.Array] = 18,
    cutoff: float = 0,
):
    _, total_wavenumber = grid.modal_axes
    k = total_wavenumber / total_wavenumber.max()
    a = attenuation
    c = cutoff
    p = order
    scaling = jnp.exp((k > c) * (-a * (((k - c) / (1 - c))**(2 * p))))
    return _make_filter_fn(scaling, "exponential_filter")


def horizontal_diffusion_filter(
    grid: spherical_harmonic.Grid,
    scale: Union[float, typing.Array],
    order: int = 1,
) -> Callable[[typing.PyTreeState], typing.PyTreeState]:
    eigenvalues = grid.laplacian_eigenvalues
    scaling = jnp.exp(-scale * (-eigenvalues)**order)
    return _make_filter_fn(scaling, "horizontal_diffusion_filter")
