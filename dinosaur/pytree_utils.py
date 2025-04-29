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
    f,
    x,
    *,
    scalar_fn=lambda x: x,
    backend: str = 'jax',
):
    as_array_fn = {'jax': jnp.asarray, 'numpy': np.asarray}[backend]

    def g(x):
        x = as_array_fn(x)
        return f(x) if x.ndim else scalar_fn(x)

    return tree_map(g, x)


def as_dict(inputs):
    return_type = type(inputs)
    inputs = inputs.asdict()
    from_dict_fn = lambda dict_inputs: return_type(**dict_inputs)
    return inputs, from_dict_fn
