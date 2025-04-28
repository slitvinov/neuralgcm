import dataclasses
import functools
from typing import Any, Callable, Mapping, MutableMapping, Sequence, TypeVar, Union
from dinosaur import coordinate_systems
from dinosaur import primitive_equations
from dinosaur import scales
from dinosaur import sigma_coordinates
from dinosaur import spherical_harmonic
from dinosaur import typing
from dinosaur import vertical_interpolation
import fsspec
import jax
import numpy as np
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
XR_REALIZATION_NAME = 'realization'
OROGRAPHY = 'orography'
GEOPOTENTIAL_KEY = 'geopotential'
REF_TEMP_KEY = 'ref_temperatures'
REF_POTENTIAL_KEY = 'reference_potential'
REFERENCE_DATETIME_KEY = 'reference_datetime'
XARRAY_DS_KEY = 'xarray_dataset'
NODAL_AXES_NAMES = (
    XR_LON_NAME,
    XR_LAT_NAME,
)
MODAL_AXES_NAMES = (
    XR_LON_MODE_NAME,
    XR_LAT_MODE_NAME,
)
Grid = spherical_harmonic.Grid


def _maybe_update_shape_and_dim_with_realization_time_sample(
    shape,
    dims,
    times,
    sample_ids,
    include_realization,
):
    not_scalar = bool(shape)
    if times is not None:
        shape = times.shape + shape
        dims = (XR_TIME_NAME, ) + dims
    return shape, dims


def _infer_dims_shape_and_coords(
    coords,
    times,
    sample_ids,
    additional_coords,
):
    lon_k, lat_k = coords.horizontal.modal_axes
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
    basic_shape_to_dims[tuple()] = tuple()
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
            continue
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
    return all_xr_coords, shape_to_dims


def data_to_xarray(
    data,
    *,
    coords,
    times,
    sample_ids=None,
    additional_coords=None,
    attrs=None,
    serialize_coords_to_attrs=True,
):
    prognostic_keys = set(data.keys()) - {'tracers'} - {'diagnostics'}
    tracer_keys = data['tracers'].keys() if 'tracers' in data else set()
    diagnostic_keys = (data['diagnostics'].keys()
                       if 'diagnostics' in data else set())
    if additional_coords is None:
        additional_coords = {}
    if (coords.vertical.layers != 1) and (XR_SURFACE_NAME
                                          not in additional_coords):
        additional_coords[XR_SURFACE_NAME] = np.ones(1)
    all_coords, shape_to_dims = _infer_dims_shape_and_coords(
        coords, times, sample_ids, additional_coords)
    dims_in_state = set()
    data_vars = {}
    for key in prognostic_keys:
        value = data[key]
        dims = shape_to_dims[value.shape]
        data_vars[key] = (dims, value)
        dims_in_state.update(set(dims))
    dataset_attrs = coords.asdict() if serialize_coords_to_attrs else {}
    coords = {k: v for k, v in all_coords.items() if k in dims_in_state}
    return xarray.Dataset(data_vars, coords, attrs=dataset_attrs)


def temperature_variation_to_absolute(
    temperature_variation,
    ref_temperature,
):
    return temperature_variation + ref_temperature[:, np.newaxis, np.newaxis]
