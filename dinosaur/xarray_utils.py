import dataclasses
import functools
from typing import Any, Callable, Mapping, MutableMapping, Sequence, TypeVar, Union
import dask.array
from dinosaur import coordinate_systems
from dinosaur import horizontal_interpolation
from dinosaur import layer_coordinates
from dinosaur import primitive_equations
from dinosaur import scales
from dinosaur import sigma_coordinates
from dinosaur import spherical_harmonic
from dinosaur import typing
from dinosaur import vertical_interpolation
from dinosaur import weatherbench_utils
import fsspec
import jax
import numpy as np
import pandas as pd
from sklearn import neighbors
import xarray

XR_SAMPLE_NAME = 'sample'
XR_TIME_NAME = 'time'
XR_INIT_TIME_NAME = 'initial_time'
XR_TIMEDELTA_NAME = 'prediction_timedelta'
XR_LEVEL_NAME = 'level'
XR_SURFACE_NAME = 'surface'
XR_LON_NAME = 'lon'
XR_LAT_NAME = 'lat'
XR_LON_MODE_NAME = 'longitudinal_mode'
XR_LAT_MODE_NAME = 'total_wavenumber'
XR_AUX_FEATURES_LIST_KEY = 'aux_features_key'
XR_REALIZATION_NAME = 'realization'
OROGRAPHY = 'orography'  # key for referring to orography.
GEOPOTENTIAL_KEY = 'geopotential'  # key for referring to geopotential.
GEOPOTENTIAL_AT_SURFACE_KEY = 'geopotential_at_surface'
REF_TEMP_KEY = 'ref_temperatures'  # key for referring to reference temperature.
REF_POTENTIAL_KEY = 'reference_potential'  # key for referring to ref potential.
REFERENCE_DATETIME_KEY = 'reference_datetime'  # key for referencing 0-time.
XARRAY_DS_KEY = 'xarray_dataset'
GEOPOTENTIAL_AT_SURFACE = (  # Geopotential, z, m**2 s**-2
    'geopotential_at_surface')
HIGH_VEGETATION_COVER = (  # High vegetation cover, cvh, (0 - 1)
    'high_vegetation_cover')
LAKE_COVER = 'lake_cover'  # Lake cover, cl, (0 - 1)
LAKE_DEPTH = 'lake_depth'  # Lake total depth, dl, m
LAND_SEA_MASK = 'land_sea_mask'  # Land-sea mask, lsm, (0 - 1)
LOW_VEGETATION_COVER = (  # Low vegetation cover, cvl, (0 - 1)
    'low_vegetation_cover')
SOIL_TYPE = 'soil_type'  # Soil type, slt, ~
TYPE_OF_HIGH_VEGETATION = (  # Type of high vegetation, tvh, ~
    'type_of_high_vegetation')
TYPE_OF_LOW_VEGETATION = (  # Type of low vegetation, tvl, ~
    'type_of_low_vegetation')
single_level_static_vars = [
    GEOPOTENTIAL_AT_SURFACE,
    HIGH_VEGETATION_COVER,
    LAKE_COVER,
    LAKE_DEPTH,
    LAND_SEA_MASK,
    LOW_VEGETATION_COVER,
    SOIL_TYPE,
    TYPE_OF_HIGH_VEGETATION,
    TYPE_OF_LOW_VEGETATION,
]
ICE_TEMPERATURE_LAYER_4 = (  # Ice temperature layer 4, istl4, K
    'ice_temperature_layer_4')
LAKE_ICE_DEPTH = 'lake_ice_depth'  # Lake ice total depth, licd, m
LAKE_ICE_TEMPERATURE = (  # Lake ice surface temperature, lict, K
    'lake_ice_temperature')
SEA_ICE_COVER = 'sea_ice_cover'  # Sea ice area fraction, siconc, (0 - 1)
SEA_SURFACE_TEMPERATURE = (  # Sea surface temperature, sst, K
    'sea_surface_temperature')
SNOW_DEPTH = 'snow_depth'  # Snow depth, sd, m of water equivalent
SOIL_TEMPERATURE_LEVEL_4 = (  # Soil temperature level 4, stl4, K
    'soil_temperature_level_4')
VOLUMETRIC_SOIL_WATER_LAYER_4 = (  # Volumetric soil water layer 4, swvl4, m**3 m**-3
    'volumetric_soil_water_layer_4')
single_level_dynamic_vars = [
    ICE_TEMPERATURE_LAYER_4,
    LAKE_ICE_DEPTH,
    LAKE_ICE_TEMPERATURE,
    SEA_ICE_COVER,
    SEA_SURFACE_TEMPERATURE,
    SNOW_DEPTH,
    SOIL_TEMPERATURE_LEVEL_4,
    VOLUMETRIC_SOIL_WATER_LAYER_4,
]
NODAL_AXES_NAMES = (
    XR_LON_NAME,
    XR_LAT_NAME,
)
MODAL_AXES_NAMES = (
    XR_LON_MODE_NAME,
    XR_LAT_MODE_NAME,
)
GRID_REGISTRY = {
    'SigmaCoordinates':
    sigma_coordinates.SigmaCoordinates,
    'LayerCoordinates':
    layer_coordinates.LayerCoordinates,
    'PressureCoordinates':
    vertical_interpolation.PressureCoordinates,
    'Grid':
    spherical_harmonic.Grid,
    'RealSphericalHarmonics':
    spherical_harmonic.RealSphericalHarmonics,
    'RealSphericalHarmonicsWithZeroImag':
    (spherical_harmonic.RealSphericalHarmonicsWithZeroImag),
    'FastSphericalHarmonics':
    spherical_harmonic.FastSphericalHarmonics,
}
LINEAR = 'LINEAR'
CUBIC = 'CUBIC'
Grid = spherical_harmonic.Grid
CUBIC_SHAPE_TO_GRID_DICT = {
    Grid.T21().nodal_shape: Grid.T21,
    Grid.T31().nodal_shape: Grid.T31,
    Grid.T42().nodal_shape: Grid.T42,
    Grid.T85().nodal_shape: Grid.T85,
    Grid.T106().nodal_shape: Grid.T106,
    Grid.T119().nodal_shape: Grid.T119,
    Grid.T170().nodal_shape: Grid.T170,
    Grid.T213().nodal_shape: Grid.T213,
    Grid.T340().nodal_shape: Grid.T340,
    Grid.T425().nodal_shape: Grid.T425,
}
LINEAR_SHAPE_TO_GRID_DICT = {
    Grid.TL31().nodal_shape: Grid.TL31,
    Grid.TL47().nodal_shape: Grid.TL47,
    Grid.TL63().nodal_shape: Grid.TL63,
    Grid.TL95().nodal_shape: Grid.TL95,
    Grid.TL127().nodal_shape: Grid.TL127,
    Grid.TL159().nodal_shape: Grid.TL159,
    Grid.TL179().nodal_shape: Grid.TL179,
    Grid.TL255().nodal_shape: Grid.TL255,
    Grid.TL639().nodal_shape: Grid.TL639,
    Grid.TL1279().nodal_shape: Grid.TL1279,
}


def is_dir(path: str) -> bool:
    protocol, path = fsspec.core.split_protocol(path)
    fs = fsspec.filesystem(protocol=protocol)
    return fs.isdir(path)


def open_dataset(
    path: str,
    **kwargs,
) -> xarray.Dataset:  # pylint: disable=redefined-builtin
    if is_dir(path):
        return xarray_tensorstore.open_zarr(path, **kwargs)
    else:
        return open_netcdf(path, **kwargs)


def open_netcdf(path: str,
                max_parallel_reads: Union[int, None] = None,
                **kwargs) -> xarray.Dataset:
    del max_parallel_reads  # unused.
    with fsspec.open(path, 'rb') as f:
        return xarray.load_dataset(f.read(), **kwargs)


def save_netcdf(dataset: xarray.Dataset, path: str):
    with fsspec.open(path, 'wb') as f:
        f.write(dataset.to_netcdf())


def _maybe_update_shape_and_dim_with_realization_time_sample(
    shape: tuple[int, ...],
    dims: tuple[str, ...],
    times: typing.Array,
    sample_ids: typing.Array,
    include_realization: bool,
) -> tuple[Sequence[int], Sequence[str]]:
    not_scalar = bool(shape)
    if times is not None:
        shape = times.shape + shape
        dims = (XR_TIME_NAME, ) + dims
    if sample_ids is not None:
        shape = sample_ids.shape + shape
        dims = (XR_SAMPLE_NAME, ) + dims
    if not_scalar and include_realization:
        shape = (1, ) + shape
        dims = (XR_REALIZATION_NAME, ) + dims
    return shape, dims


def _infer_dims_shape_and_coords(
    coords: coordinate_systems.CoordinateSystem,
    times: Union[typing.Array, None],
    sample_ids: typing.Array,
    additional_coords: Mapping[str, typing.Array],
) -> tuple[dict[str, typing.Array], dict[tuple[int, ...], tuple[int, ...]]]:
    lon_k, lat_k = coords.horizontal.modal_axes  # k stands for wavenumbers
    lon, sin_lat = coords.horizontal.nodal_axes
    all_xr_coords = {
        XR_LON_NAME: lon * 180 / np.pi,
        XR_LAT_NAME: np.arcsin(sin_lat) * 180 / np.pi,
        XR_LON_MODE_NAME: lon_k,
        XR_LAT_MODE_NAME: lat_k,
        XR_LEVEL_NAME: coords.vertical.centers,
        **additional_coords,
    }
    if times is not None:
        all_xr_coords[XR_TIME_NAME] = times
    if sample_ids is not None:
        all_xr_coords[XR_SAMPLE_NAME] = sample_ids
    basic_shape_to_dims = {}
    basic_shape_to_dims[tuple()] = tuple()  # scalar variables
    modal_shape = coords.horizontal.modal_shape
    nodal_shape = coords.horizontal.nodal_shape
    basic_shape_to_dims[(coords.vertical.layers, ) +
                        modal_shape] = (XR_LEVEL_NAME, ) + MODAL_AXES_NAMES
    basic_shape_to_dims[(coords.vertical.layers, ) +
                        nodal_shape] = (XR_LEVEL_NAME, ) + NODAL_AXES_NAMES
    basic_shape_to_dims[nodal_shape] = NODAL_AXES_NAMES
    basic_shape_to_dims[modal_shape] = MODAL_AXES_NAMES
    basic_shape_to_dims[coords.surface_nodal_shape] = NODAL_AXES_NAMES
    for dim, value in additional_coords.items():
        if dim == XR_REALIZATION_NAME:
            continue  # Handled in _maybe_update_shape_and_dim_with_time_sample
        if value.ndim != 1:
            raise ValueError(
                '`additional_coords` must be 1d vectors, but got: '
                f'{value.shape=} for {dim=}')
        if value.shape == (coords.vertical.layers, ):
            raise ValueError(
                f'`additional_coords` {dim=} has shape={value.shape} that collides '
                f'with {XR_LEVEL_NAME=}. Since matching of axes is done using shape, '
                'consider renaming after the fact.')
        basic_shape_to_dims[value.shape +
                            modal_shape] = (dim, ) + MODAL_AXES_NAMES
        basic_shape_to_dims[value.shape +
                            nodal_shape] = (dim, ) + NODAL_AXES_NAMES
        basic_shape_to_dims[value.shape] = (dim, )
    update_shape_dims_fn = functools.partial(
        _maybe_update_shape_and_dim_with_realization_time_sample,
        times=times,
        sample_ids=sample_ids,
        include_realization=XR_REALIZATION_NAME in additional_coords,
    )
    shape_to_dims = {}
    for shape, dims in basic_shape_to_dims.items():
        full_shape, full_dims = update_shape_dims_fn(shape, dims)
        shape_to_dims[full_shape] = full_dims
    return all_xr_coords, shape_to_dims  # pytype: disable=bad-return-type


def nodal_orography_from_ds(ds: xarray.Dataset) -> typing.Array:
    orography_key = OROGRAPHY
    if orography_key not in ds:
        ds[orography_key] = (ds[GEOPOTENTIAL_AT_SURFACE_KEY] /
                             scales.GRAVITY_ACCELERATION.magnitude)
    lon_lat_order = (XR_LON_NAME, XR_LAT_NAME)
    return ds[orography_key].transpose(*lon_lat_order).values


def nodal_land_sea_mask_from_ds(ds: xarray.Dataset) -> typing.Array:
    land_sea_mask_key = LAND_SEA_MASK
    lon_lat_order = ('longitude', 'latitude')
    return ds[land_sea_mask_key].transpose(*lon_lat_order).values


def data_to_xarray(
    data: dict,
    *,
    coords: coordinate_systems.CoordinateSystem,
    times: Union[typing.Array, None],
    sample_ids: Union[typing.Array, None] = None,
    additional_coords: Union[MutableMapping[str, typing.Array], None] = None,
    attrs: Union[Mapping[str, Any], None] = None,
    serialize_coords_to_attrs: bool = True,
) -> xarray.Dataset:
    prognostic_keys = set(data.keys()) - {'tracers'} - {'diagnostics'}
    tracer_keys = data['tracers'].keys() if 'tracers' in data else set()
    diagnostic_keys = (data['diagnostics'].keys()
                       if 'diagnostics' in data else set())
    if not prognostic_keys.isdisjoint(tracer_keys):
        raise ValueError(
            'Tracer names collide with prognostic variables',
            f'Tracers: {tracer_keys}; prognostics: {prognostic_keys}',
        )
    if not prognostic_keys.isdisjoint(diagnostic_keys):
        raise ValueError(
            'Diagnostic names collide with prognostic variables',
            f'Diagnostic: {diagnostic_keys}; ',
            f'prognostics: {prognostic_keys}',
        )
    if additional_coords is None:
        additional_coords = {}
    if (coords.vertical.layers != 1) and (XR_SURFACE_NAME
                                          not in additional_coords):
        additional_coords[XR_SURFACE_NAME] = np.ones(1)
    all_coords, shape_to_dims = _infer_dims_shape_and_coords(
        coords, times, sample_ids, additional_coords)
    dims_in_state = set()  # keep track which coordinates should be included.
    data_vars = {}
    for key in prognostic_keys:
        value = data[key]
        if value.shape not in shape_to_dims:
            raise ValueError(
                f'Value of shape {value.shape} is not in {shape_to_dims=}')
        else:
            dims = shape_to_dims[value.shape]
            data_vars[key] = (dims, value)
            dims_in_state.update(set(dims))
    for key in tracer_keys:
        value = data['tracers'][key]
        if value.shape not in shape_to_dims:
            raise ValueError(
                f'Value of shape {value.shape} is not recognized.')
        else:
            dims = shape_to_dims[value.shape]
            data_vars[key] = (dims, value)
            dims_in_state.update(set(dims))
    for key in diagnostic_keys:
        value = data['diagnostics'][key]
        if value.shape not in shape_to_dims:
            raise ValueError(
                f'Value of shape {value.shape} is not recognized.')
        else:
            dims = shape_to_dims[value.shape]
            data_vars[key] = (dims, value)
            dims_in_state.update(set(dims))
    dataset_attrs = coords.asdict() if serialize_coords_to_attrs else {}
    if attrs is not None:
        for key in dataset_attrs.keys():
            if key in attrs:
                raise ValueError(f'Key {key} is not allowed in `attrs`.')
        dataset_attrs.update(attrs)
    coords = {k: v for k, v in all_coords.items() if k in dims_in_state}
    return xarray.Dataset(data_vars, coords, attrs=dataset_attrs)


def temperature_variation_to_absolute(
    temperature_variation: np.ndarray,
    ref_temperature: np.ndarray,
) -> np.ndarray:
    return temperature_variation + ref_temperature[:, np.newaxis, np.newaxis]
