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
def _single_device_dot_cumsum(x: jax.Array, axis: int, reverse: bool = False):
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


def _dot_cumsum(
    x: jax.Array,
    axis: int,
    sharding: jax.sharding.NamedSharding | None,
    reverse: bool = False,
):
    return _single_device_dot_cumsum(x, axis, reverse=reverse)


def cumsum(
    x: np.ndarray | jax.Array,
    axis: int,
    method: str = 'dot',
    sharding: jax.sharding.NamedSharding | None = None,
):
    return _dot_cumsum(x, axis, sharding=sharding)


def pad_in_dim(x: np.ndarray | jax.Array, pad_width: tuple[int, int],
               axis: int):
    padding_value = jnp.array(0, dtype=x.dtype)
    padding_config = [(0, 0, 0)] * x.ndim
    padding_config[axis] = pad_width + (0,
                                        )  # add "interior" padding for lax.pad
    return lax.pad(x, padding_value, padding_config)


def shift(x: np.ndarray | jax.Array, offset: int, axis: int):
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
