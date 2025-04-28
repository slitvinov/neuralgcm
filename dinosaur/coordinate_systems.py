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
    if len(sharding.spec) != 3:
        raise ValueError(
            f'partition spec does not have length 3: {sharding.spec}')

    def f(y: jax.Array) -> jax.Array:
        if y.ndim == 1 and y.dtype == jnp.uint32:
            return y  # prng key
        if y.ndim not in {2, 3}:
            raise ValueError(f'can only shard 2D or 3D arrays: {y.shape=}')
        if y.ndim == 2:
            spec = P(*sharding.spec[1:])
            sharding_ = jax.sharding.NamedSharding(sharding.mesh, spec)
        elif y.shape[0] == 1:
            spec = P(None, *sharding.spec[1:])
            sharding_ = jax.sharding.NamedSharding(sharding.mesh, spec)
        else:
            sharding_ = sharding
        return jax.lax.with_sharding_constraint(y, sharding_)

    try:
        return pytree_utils.tree_map_over_nonscalars(f, x)
    except ValueError as e:
        shapes = jax.tree_util.tree_map(jnp.shape, x)
        raise ValueError(f'failed to shard pytree with shapes {shapes}') from e


@dataclasses.dataclass(frozen=True)
class CoordinateSystem:
    horizontal: HorizontalGridTypes
    vertical: VerticalCoordinateTypes
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


def get_spectral_downsample_fn(
    coords: CoordinateSystem,
    save_coords: CoordinateSystem,
    expect_same_vertical: bool = True,
) -> Callable[[typing.PyTreeState], typing.PyTreeState]:
    if expect_same_vertical and (coords.vertical != save_coords.vertical):
        raise ValueError('downsampling vertical resolution is not supported.')
    lon_wavenumber_slice = slice(0, save_coords.horizontal.modal_shape[0])
    total_wavenumber_slice = slice(0, save_coords.horizontal.modal_shape[1])
    if (coords.horizontal.total_wavenumbers
            < save_coords.horizontal.total_wavenumbers) or (
                coords.horizontal.longitude_wavenumbers
                < save_coords.horizontal.longitude_wavenumbers):
        raise ValueError(
            'save_coords.horizontal larger than coords.horizontal')

    def downsample_fn(state: typing.PyTreeState) -> typing.PyTreeState:
        slice_fn = lambda x: x[..., lon_wavenumber_slice,
                               total_wavenumber_slice]
        return pytree_utils.tree_map_over_nonscalars(slice_fn, state)

    return downsample_fn


def get_spectral_upsample_fn(
    coords: CoordinateSystem,
    save_coords: CoordinateSystem,
    expect_same_vertical: bool = True,
) -> Callable[[typing.PyTreeState], typing.PyTreeState]:
    if expect_same_vertical and (coords.vertical != save_coords.vertical):
        raise ValueError('upsampling vertical resolution is not supported.')
    save_shape = save_coords.horizontal.modal_shape
    coords_shape = coords.horizontal.modal_shape
    lon_wavenumber_pad = (0, save_shape[0] - coords_shape[0])
    total_wavenumber_pad = (0, save_shape[1] - coords_shape[1])
    if (min(lon_wavenumber_pad) != 0) or (min(total_wavenumber_pad) != 0):
        raise ValueError(
            'save_coords.horizontal smaller than coords.horizontal')
    tail_pad = (lon_wavenumber_pad, total_wavenumber_pad)

    def upsample_fn(state: typing.PyTreeState) -> typing.PyTreeState:
        pad_fn = lambda x: jnp.pad(x, ((0, 0), ) * (x.ndim - 2) + tail_pad)
        return pytree_utils.tree_map_over_nonscalars(pad_fn, state)

    return upsample_fn


def get_spectral_interpolate_fn(
    source_coords: CoordinateSystem,
    target_coords: CoordinateSystem,
    expect_same_vertical: bool = True,
) -> Callable[[typing.PyTreeState], typing.PyTreeState]:
    if (source_coords.horizontal.total_wavenumbers
            < target_coords.horizontal.total_wavenumbers) and (
                source_coords.horizontal.longitude_wavenumbers
                < target_coords.horizontal.longitude_wavenumbers):
        return get_spectral_upsample_fn(source_coords, target_coords,
                                        expect_same_vertical)
    elif (source_coords.horizontal.total_wavenumbers
          >= target_coords.horizontal.total_wavenumbers) and (
              source_coords.horizontal.longitude_wavenumbers
              >= target_coords.horizontal.longitude_wavenumbers):
        return get_spectral_downsample_fn(source_coords, target_coords,
                                          expect_same_vertical)
    else:
        raise ValueError('Incompatible horizontal coordinates with shapes '
                         f'{source_coords.horizontal.modal_shape}, '
                         f'{target_coords.horizontal.modal_shape}')


