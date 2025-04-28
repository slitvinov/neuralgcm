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

NODAL_AXES_NAMES = (
    'lon',
    'lat',
)
MODAL_AXES_NAMES = (
    'longitudinal_mode',
    'total_wavenumber',
)


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
        dims = ('time', ) + dims
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
        'lon': lon * 180 / np.pi,
        'lat': np.arcsin(sin_lat) * 180 / np.pi,
        'longitudinal_mode': lon_k,
        'total_wavenumber': lat_k,
        'level': coords.vertical.centers,
        **additional_coords,
    }
    if times is not None:
        all_xr_coords['time'] = times
    if sample_ids is not None:
        all_xr_coords['sample'] = sample_ids
    basic_shape_to_dims = {}
    basic_shape_to_dims[tuple()] = tuple()
    modal_shape = coords.horizontal.modal_shape
    nodal_shape = coords.horizontal.nodal_shape
    basic_shape_to_dims[(coords.vertical.layers, ) +
                        modal_shape] = ('level', ) + MODAL_AXES_NAMES
    basic_shape_to_dims[(coords.vertical.layers, ) +
                        nodal_shape] = ('level', ) + NODAL_AXES_NAMES
    basic_shape_to_dims[nodal_shape] = NODAL_AXES_NAMES
    basic_shape_to_dims[modal_shape] = MODAL_AXES_NAMES
    basic_shape_to_dims[coords.surface_nodal_shape] = NODAL_AXES_NAMES
    for dim, value in additional_coords.items():
        if dim == 'realization':
            continue
        basic_shape_to_dims[value.shape +
                            modal_shape] = (dim, ) + MODAL_AXES_NAMES
        basic_shape_to_dims[value.shape +
                            nodal_shape] = (dim, ) + NODAL_AXES_NAMES
        basic_shape_to_dims[value.shape] = (dim, )
    update_shape_dims_fn = functools.partial(
        _maybe_update_shape_and_dim_with_realization_time_sample,
        times=times,
        sample_ids=sample_ids,
        include_realization='realization' in additional_coords,
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
    if (coords.vertical.layers != 1) and ('surface' not in additional_coords):
        additional_coords['surface'] = np.ones(1)
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
