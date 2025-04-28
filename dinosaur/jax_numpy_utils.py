from __future__ import annotations
import functools
import math
import re
import jax
from jax import lax
from jax.experimental import shard_map
import jax.numpy as jnp
import numpy as np


@jax.named_call
def _single_device_dot_cumsum(x: jax.Array,
                              axis: int,
                              reverse: bool = False) -> jax.Array:
    if not -x.ndim <= axis < x.ndim:
        raise ValueError(f'invalid {axis=}')
    if axis < 0:
        axis = axis + x.ndim
    size = x.shape[axis]
    i = jnp.arange(size)[:, jnp.newaxis]
    j = jnp.arange(size)[jnp.newaxis, :]
    op = jnp.greater_equal if reverse else jnp.less_equal
    w = op(i, j).astype(np.float32)
    out_axes = list(range(x.ndim))
    out_axes[axis] = x.ndim
    return jnp.einsum(
        w,
        [axis, x.ndim],
        x,
        list(range(x.ndim)),
        out_axes,
        precision=('bfloat16', 'highest'),
    )


def _parallel_dot_cumsum(x: jax.Array, axis: int, reverse: bool,
                         axis_name: str) -> jax.Array:
    partials = _single_device_dot_cumsum(x, axis=axis, reverse=reverse)
    last_partial = lax.index_in_dim(partials, 0 if reverse else -1, axis)
    sums = lax.all_gather(last_partial, axis_name, tiled=True)
    axis_index = lax.axis_index(axis_name)
    op = jnp.greater if reverse else jnp.less
    total = partials
    terms = sums[1:] if reverse else sums[:-1]
    start = 1 if reverse else 0
    for i, term in enumerate(terms, start=start):
        total += op(i, axis_index) * term
    return total


def _dot_cumsum(
    x: jax.Array,
    axis: int,
    sharding: jax.sharding.NamedSharding | None,
    reverse: bool = False,
) -> jax.Array:
    if sharding is None or sharding.spec[axis] is None:
        return _single_device_dot_cumsum(x, axis, reverse=reverse)
    mesh = sharding.mesh
    spec = sharding.spec

    @jax.jit
    @functools.partial(
        shard_map.shard_map,
        mesh=mesh,
        in_specs=(spec, ),
        out_specs=spec,
        check_rep=False,
    )
    def dot_cumsum(x):
        return _parallel_dot_cumsum(x,
                                    axis=axis,
                                    reverse=reverse,
                                    axis_name=sharding.spec[axis])

    return dot_cumsum(x)


def cumsum(
    x: np.ndarray | jax.Array,
    axis: int,
    method: str = 'dot',
    sharding: jax.sharding.NamedSharding | None = None,
) -> jax.Array:
    if method == 'dot':
        return _dot_cumsum(x, axis, sharding=sharding)
    elif method == 'jax':
        return jnp.cumsum(x, axis)
    else:
        raise ValueError(f'invalid {method=}')


def pad_in_dim(x: np.ndarray | jax.Array, pad_width: tuple[int, int],
               axis: int) -> jax.Array:
    padding_value = jnp.array(0, dtype=x.dtype)
    padding_config = [(0, 0, 0)] * x.ndim
    padding_config[axis] = pad_width + (0,
                                        )  # add "interior" padding for lax.pad
    return lax.pad(x, padding_value, padding_config)


def shift(x: np.ndarray | jax.Array, offset: int, axis: int) -> jax.Array:
    if abs(offset) >= x.shape[axis]:
        return jnp.zeros_like(x)
    if offset > 0:
        sliced = lax.slice_in_dim(x, 0, x.shape[axis] - offset, axis=axis)
        return pad_in_dim(sliced, (offset, 0), axis=axis)
    else:
        sliced = lax.slice_in_dim(x, -offset, x.shape[axis], axis=axis)
        return pad_in_dim(sliced, (0, -offset), axis=axis)


def diff(x, axis=-1):
    upper = lax.slice_in_dim(x, 1, None, axis=axis)
    lower = lax.slice_in_dim(x, 0, -1, axis=axis)
    return upper - lower

