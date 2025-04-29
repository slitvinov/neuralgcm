from __future__ import annotations
import dataclasses
from typing import Any, Union
from dinosaur import pytree_utils
import jax
import numpy as np

@dataclasses.dataclass(frozen=True)
class CoordinateSystem:
    horizontal: Any
    vertical: Any

    def asdict(self):
        out = {**self.horizontal.asdict(), **self.vertical.asdict()}
        out["horizontal_grid_type"] = type(self.horizontal).__name__
        out["vertical_grid_type"] = type(self.vertical).__name__
        return out

    @property
    def surface_nodal_shape(self):
        return (1, ) + self.horizontal.nodal_shape


def get_nodal_shapes(
    inputs,
    coords,
):
    nodal_shape = coords.horizontal.nodal_shape
    array_shape_fn = lambda x: np.asarray(x.shape[:-2] + nodal_shape)
    scalar_shape_fn = lambda x: np.array([], dtype=int)
    return pytree_utils.tree_map_over_nonscalars(array_shape_fn,
                                                 inputs,
                                                 scalar_fn=scalar_shape_fn)


def maybe_to_nodal(
    fields,
    coords,
):
    nodal_shapes = get_nodal_shapes(fields, coords)

    def to_nodal_fn(x):
        return coords.horizontal.to_nodal(x)

    fn = lambda x, nodal: x if x.shape == tuple(nodal) else to_nodal_fn(x)
    return jax.tree_util.tree_map(fn, fields, nodal_shapes)
