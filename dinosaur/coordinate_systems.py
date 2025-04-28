from __future__ import annotations
import dataclasses
from typing import Any, Callable, Sequence, Union
from dinosaur import layer_coordinates
from dinosaur import pytree_utils
from dinosaur import sigma_coordinates
from dinosaur import spherical_harmonic
from dinosaur import typing
import jax
import jax.numpy as jnp
import numpy as np

HORIZONTAL_COORD_TYPE_KEY = 'horizontal_grid_type'
VERTICAL_COORD_TYPE_KEY = 'vertical_grid_type'
HorizontalGridTypes = spherical_harmonic.Grid
VerticalCoordinateTypes = Union[layer_coordinates.LayerCoordinates,
                                sigma_coordinates.SigmaCoordinates, Any]
P = jax.sharding.PartitionSpec


def _with_sharding_constraint(
    x: typing.Pytree,
    sharding: Union[jax.sharding.NamedSharding, None],
) -> typing.Pytree:
    if sharding is None:
        return x  # unsharded
    assert False


@dataclasses.dataclass(frozen=True)
class CoordinateSystem:
    horizontal: Any
    vertical: Any
    spmd_mesh: Union[jax.sharding.Mesh, None] = None
    dycore_partition_spec: jax.sharding.PartitionSpec = P('z', 'x', 'y')
    physics_partition_spec: jax.sharding.PartitionSpec = P(
        None, ('x', 'z'), 'y')

    def __post_init__(self):
        if self.spmd_mesh is not None:
            if not {'x', 'y', 'z'} <= set(self.spmd_mesh.axis_names):
                raise ValueError(
                    "mesh is missing one or more of the required axis names 'x', 'y' "
                    f"and 'z': {self.spmd_mesh}")
        horizontal = dataclasses.replace(self.horizontal,
                                         spmd_mesh=self.spmd_mesh)
        object.__setattr__(self, 'horizontal', horizontal)

    def _get_sharding(
        self, partition_spec: jax.sharding.PartitionSpec
    ) -> Union[jax.sharding.NamedSharding, None]:
        if self.spmd_mesh is None:
            return None
        return jax.sharding.NamedSharding(self.spmd_mesh, partition_spec)

    @property
    def physics_sharding(self) -> Union[jax.sharding.NamedSharding, None]:
        return self._get_sharding(self.physics_partition_spec)

    def with_physics_sharding(self,
                              x: typing.PyTreeState) -> typing.PyTreeState:
        return _with_sharding_constraint(x, self.physics_sharding)

    @property
    def dycore_sharding(self) -> Union[jax.sharding.NamedSharding, None]:
        return self._get_sharding(self.dycore_partition_spec)

    def with_dycore_sharding(self,
                             x: typing.PyTreeState) -> typing.PyTreeState:
        return _with_sharding_constraint(x, self.dycore_sharding)

    def dycore_to_physics_sharding(
            self, x: typing.PyTreeState) -> typing.PyTreeState:
        return self.with_physics_sharding(self.with_dycore_sharding(x))

    def physics_to_dycore_sharding(
            self, x: typing.PyTreeState) -> typing.PyTreeState:
        return self.with_dycore_sharding(self.with_physics_sharding(x))

    def asdict(self) ->...:
        horizontal_keys = set(self.horizontal.asdict().keys())
        vertical_keys = set(self.vertical.asdict().keys())
        if horizontal_keys.intersection(vertical_keys):
            raise ValueError('keys in horizontal and vertical grids collide.')
        out = {**self.horizontal.asdict(), **self.vertical.asdict()}
        out[HORIZONTAL_COORD_TYPE_KEY] = type(self.horizontal).__name__
        out[VERTICAL_COORD_TYPE_KEY] = type(self.vertical).__name__
        return out

    @property
    def nodal_shape(self) -> tuple[int, int, int]:
        return (self.vertical.layers, ) + self.horizontal.nodal_shape

    @property
    def modal_shape(self) -> tuple[int, int, int]:
        return (self.vertical.layers, ) + self.horizontal.modal_shape

    @property
    def surface_nodal_shape(self) -> tuple[int, int, int]:
        return (1, ) + self.horizontal.nodal_shape

    @property
    def surface_modal_shape(self) -> tuple[int, int, int]:
        return (1, ) + self.horizontal.modal_shape


def get_nodal_shapes(
    inputs: typing.Pytree,
    coords: CoordinateSystem,
) -> typing.Pytree:
    nodal_shape = coords.horizontal.nodal_shape
    array_shape_fn = lambda x: np.asarray(x.shape[:-2] + nodal_shape)
    scalar_shape_fn = lambda x: np.array([], dtype=int)
    return pytree_utils.tree_map_over_nonscalars(array_shape_fn,
                                                 inputs,
                                                 scalar_fn=scalar_shape_fn)


def maybe_to_nodal(
    fields: typing.Pytree,
    coords: CoordinateSystem,
) -> typing.Pytree:
    nodal_shapes = get_nodal_shapes(fields, coords)

    def to_nodal_fn(x):
        return coords.with_dycore_sharding(
            coords.horizontal.to_nodal(coords.with_dycore_sharding(x)))

    fn = lambda x, nodal: x if x.shape == tuple(nodal) else to_nodal_fn(x)
    return jax.tree_util.tree_map(fn, fields, nodal_shapes)
