from collections import abc
import dataclasses
import functools
from typing import Any, Callable, Sequence, Union
from dinosaur import typing
import jax
import jax.numpy as jnp
import numpy as np

tree_map = jax.tree_util.tree_map

def tree_map_over_nonscalars(
    f: Callable[[typing.Array], typing.Array],
    x: typing.Pytree,
    *,
    scalar_fn: Callable[[typing.Array], typing.Array] = lambda x: x,
    backend: str = 'jax',
) -> typing.Pytree:
    as_array_fn = {'jax': jnp.asarray, 'numpy': np.asarray}[backend]

    def g(x: typing.Array) -> typing.Array:
        x = as_array_fn(x)
        return f(x) if x.ndim else scalar_fn(x)

    return tree_map(g, x)

def as_dict(inputs: typing.Pytree) -> typing.Pytree:
    return_type = type(inputs)
    if dataclasses.is_dataclass(inputs):
        inputs = inputs.asdict()
    else:
        assert 0
    from_dict_fn = lambda dict_inputs: return_type(**dict_inputs)
    return inputs, from_dict_fn

