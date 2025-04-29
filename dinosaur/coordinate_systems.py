from __future__ import annotations
import dataclasses
from typing import Any, Union
from dinosaur import pytree_utils
from dinosaur import sigma_coordinates
from dinosaur import spherical_harmonic
from dinosaur import typing
import jax
import jax.numpy as jnp
import numpy as np

P = jax.sharding.PartitionSpec


def _with_sharding_constraint(x, sharding):
    return x


@dataclasses.dataclass(frozen=True)
class CoordinateSystem:
    horizontal: Any
    vertical: Any
    spmd_mesh: Union[jax.sharding.Mesh, None] = None
    dycore_partition_spec: jax.sharding.PartitionSpec = P('z', 'x', 'y')
    physics_partition_spec: jax.sharding.PartitionSpec = P(
        None, ('x', 'z'), 'y')

    def __post_init__(self):
        horizontal = dataclasses.replace(self.horizontal,
                                         spmd_mesh=self.spmd_mesh)
        object.__setattr__(self, 'horizontal', horizontal)

    def _get_sharding(self, partition_spec: jax.sharding.PartitionSpec):
        return None

    @property
    def dycore_sharding(self):
        return self._get_sharding(self.dycore_partition_spec)

    def with_dycore_sharding(self, x):
        return _with_sharding_constraint(x, self.dycore_sharding)

    def asdict(self) ->...:
        out = {**self.horizontal.asdict(), **self.vertical.asdict()}
        out['horizontal_grid_type'] = type(self.horizontal).__name__
        out['vertical_grid_type'] = type(self.vertical).__name__
        return out

    @property
    def surface_nodal_shape(self):
        return (1, ) + self.horizontal.nodal_shape


def get_nodal_shapes(
    inputs,
    coords: CoordinateSystem,
):
    nodal_shape = coords.horizontal.nodal_shape
    array_shape_fn = lambda x: np.asarray(x.shape[:-2] + nodal_shape)
    scalar_shape_fn = lambda x: np.array([], dtype=int)
    return pytree_utils.tree_map_over_nonscalars(array_shape_fn,
                                                 inputs,
                                                 scalar_fn=scalar_shape_fn)


def maybe_to_nodal(
    fields,
    coords: CoordinateSystem,
):
    nodal_shapes = get_nodal_shapes(fields, coords)

    def to_nodal_fn(x):
        return coords.with_dycore_sharding(
            coords.horizontal.to_nodal(coords.with_dycore_sharding(x)))

    fn = lambda x, nodal: x if x.shape == tuple(nodal) else to_nodal_fn(x)
    return jax.tree_util.tree_map(fn, fields, nodal_shapes)
