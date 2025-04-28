from typing import Callable, Union
from dinosaur import coordinate_systems
from dinosaur import filtering
from dinosaur import primitive_equations
from dinosaur import scales
from dinosaur import typing
from dinosaur import xarray_utils
import jax
import jax.numpy as jnp
import numpy as np

units = scales.units
Array = typing.Array
Quantity = typing.Quantity
QuantityOrStr = Union[Quantity, str]


def isothermal_rest_atmosphere(
    coords: coordinate_systems.CoordinateSystem,
    physics_specs: primitive_equations.PrimitiveEquationsSpecs,
    tref: QuantityOrStr = 288. * units.degK,
    p0: QuantityOrStr = 1e5 * units.pascal,
    p1: QuantityOrStr = 0. * units.pascal,
    surface_height: Union[Quantity, None] = None,
):
    lon, sin_lat = coords.horizontal.nodal_mesh
    lat = np.arcsin(sin_lat)
    tref = physics_specs.nondimensionalize(units.Quantity(tref))
    p0 = physics_specs.nondimensionalize(units.Quantity(p0))
    p1 = physics_specs.nondimensionalize(units.Quantity(p1))
    if surface_height is None:
        orography = np.zeros_like(lat)  # flat planet
    else:
        assert False

    def _get_vorticity(sigma, lon, lat):
        del sigma, lon  # unused.
        return jnp.zeros_like(lat)

    def _get_surface_pressure(lon, lat, rng_key):

        def relative_pressure(altitude_m):
            g = 9.80665  # m/s2
            cp = 1004.68506  # J/(kg·K)
            T0 = 288.16  # K
            M = 0.02896968  # kg/mol
            R0 = 8.314462618  # J/(mol·K)
            return (1 - g * altitude_m / (cp * T0))**(cp * M / R0)

        altitude_m = physics_specs.dimensionalize(orography,
                                                  units.meter).magnitude
        surface_pressure = (p0 * np.ones(coords.surface_nodal_shape) *
                            relative_pressure(altitude_m))
        keys = jax.random.split(rng_key, 2)
        lon0 = jax.random.uniform(keys[1],
                                  minval=np.pi / 2,
                                  maxval=3 * np.pi / 2)
        lat0 = jax.random.uniform(keys[0], minval=-np.pi / 4, maxval=np.pi / 4)
        stddev = np.pi / 20  # std deviation in lon, lat
        k = 4  # wavenumber in lon
        perturbation = (jnp.exp(-(lon - lon0)**2 / (2 * stddev**2)) *
                        jnp.exp(-(lat - lat0)**2 /
                                (2 * stddev**2)) * jnp.sin(k * (lon - lon0)))
        return surface_pressure + p1 * perturbation

    def random_state_fn(rng_key: jnp.ndarray) -> primitive_equations.State:
        nodal_vorticity = jnp.stack([
            _get_vorticity(sigma, lon, lat)
            for sigma in coords.vertical.centers
        ])
        modal_vorticity = coords.horizontal.to_modal(nodal_vorticity)
        nodal_surface_pressure = _get_surface_pressure(lon, lat, rng_key)
        return primitive_equations.State(
            vorticity=modal_vorticity,
            divergence=jnp.zeros_like(modal_vorticity),
            temperature_variation=jnp.zeros_like(modal_vorticity),
            log_surface_pressure=(coords.horizontal.to_modal(
                jnp.log(nodal_surface_pressure))),
        )

    aux_features = {
        'orography': orography,
        'ref_temperatures': np.full((coords.vertical.layers, ), tref)
    }
    return random_state_fn, aux_features


def steady_state_jw(
    coords: coordinate_systems.CoordinateSystem,
    physics_specs: primitive_equations.PrimitiveEquationsSpecs,
    u0: Quantity = 35. * units.m / units.s,
    p0: Quantity = 1e5 * units.pascal,
    t0: Quantity = 288. * units.degK,
    delta_t: Quantity = 4.8e5 * units.degK,
    gamma: Quantity = 0.005 * units.degK / units.m,
    sigma_tropo: float = 0.2,
    sigma0: float = 0.252,
):
    u0 = physics_specs.nondimensionalize(u0)
    t0 = physics_specs.nondimensionalize(t0)
    delta_t = physics_specs.nondimensionalize(delta_t)
    p0 = physics_specs.nondimensionalize(p0)
    gamma = physics_specs.nondimensionalize(gamma)
    a = physics_specs.radius
    g = physics_specs.g
    r_gas = physics_specs.R
    omega = physics_specs.angular_velocity

    def _get_reference_temperature(sigma):
        top_mean_t = t0 * sigma**(r_gas * gamma / g)
        if sigma < sigma_tropo:
            return top_mean_t + delta_t * (sigma_tropo - sigma)**5
        else:
            return top_mean_t

    def _get_reference_geopotential(sigma):
        top_mean_potential = (t0 * g / gamma) * (1 -
                                                 sigma**(r_gas * gamma / g))
        if sigma < sigma_tropo:
            return top_mean_potential - r_gas * delta_t * (
                (np.log(sigma / sigma_tropo) + 137 / 60) * sigma_tropo**5 -
                5 * sigma * sigma_tropo**4 + 5 * (sigma**2) *
                (sigma_tropo**3) - (10 / 3) * (sigma_tropo**2) * sigma**3 +
                (5 / 4) * sigma_tropo * sigma**4 - (sigma**5) / 5)
        else:
            return top_mean_potential

    def _get_geopotential(lat, lon, sigma):
        del lon  # unused.
        sigma_nu = (sigma - sigma0) * np.pi / 2
        return _get_reference_geopotential(
            sigma) + u0 * np.cos(sigma_nu)**1.5 * (
                ((-2 * np.sin(lat)**6 *
                  (np.cos(lat)**2 + 1 / 3) + 10 / 63) * u0 * np.cos(sigma_nu)**
                 (3 / 2)) +
                ((1.6 * (np.cos(lat)**3) *
                  (np.sin(lat)**2 + 2 / 3) - np.pi / 4) * a * omega))

    def _get_temperature_variation(lat, lon, sigma):
        del lon  # unused.
        sigma_nu = (sigma - sigma0) * np.pi / 2
        cos_𝜎ν = np.cos(sigma_nu)  # pylint: disable=invalid-name
        sin_𝜎ν = np.sin(sigma_nu)  # pylint: disable=invalid-name
        return 0.75 * (
            sigma * np.pi * u0 / r_gas) * sin_𝜎ν * np.sqrt(cos_𝜎ν) * (
                ((-2 * (np.cos(lat)**2 + 1 / 3) * np.sin(lat)**6 + 10 / 63) *
                 2 * u0 * cos_𝜎ν**(3 / 2)) +
                ((1.6 * (np.cos(lat)**3) *
                  (np.sin(lat)**2 + 2 / 3) - np.pi / 4) * a * omega))

    def _get_vorticity(lat, lon, sigma):
        del lon  # unused.
        sigma_nu = (sigma - sigma0) * np.pi / 2
        return ((-4 * u0 / a) * (np.cos(sigma_nu)**(3 / 2)) * np.sin(lat) *
                np.cos(lat) * (2 - 5 * np.sin(lat)**2))

    def _get_surface_pressure(
        lat,
        lon,
    ):
        del lon  # unused.
        return p0 * np.ones(lat.shape)[np.newaxis, ...]

    lon, sin_lat = coords.horizontal.nodal_mesh
    lat = np.arcsin(sin_lat)

    def initial_state_fn(
            rng_key: Union[jnp.ndarray,
                           None] = None) -> primitive_equations.State:
        del rng_key  # unused.
        nodal_vorticity = np.stack([
            _get_vorticity(lat, lon, sigma)
            for sigma in coords.vertical.centers
        ])
        modal_vorticity = coords.horizontal.to_modal(nodal_vorticity)
        nodal_temperature_variation = np.stack([
            _get_temperature_variation(lat, lon, sigma)
            for sigma in coords.vertical.centers
        ])
        log_nodal_surface_pressure = np.log(_get_surface_pressure(lat, lon))
        state = primitive_equations.State(
            vorticity=modal_vorticity,
            divergence=np.zeros_like(modal_vorticity),
            temperature_variation=coords.horizontal.to_modal(
                nodal_temperature_variation),
            log_surface_pressure=coords.horizontal.to_modal(
                log_nodal_surface_pressure))
        return state

    orography = _get_geopotential(lat, lon, 1.) / g
    geopotential = np.stack([
        _get_geopotential(lat, lon, sigma) for sigma in coords.vertical.centers
    ])
    reference_temperatures = np.stack([
        _get_reference_temperature(sigma) for sigma in coords.vertical.centers
    ])
    aux_features = {
        'geopotential': geopotential,
        'orography': orography,
        'ref_temperatures': reference_temperatures,
    }
    return initial_state_fn, aux_features


def baroclinic_perturbation_jw(
    coords: coordinate_systems.CoordinateSystem,
    physics_specs: primitive_equations.PrimitiveEquationsSpecs,
    u_perturb: Quantity = 1. * units.m / units.s,
    lon_location: Quantity = np.pi / 9,
    lat_location: Quantity = 2 * np.pi / 9,
    perturbation_radius: Quantity = 0.1,
) -> primitive_equations.State:
    u_p = physics_specs.nondimensionalize(u_perturb)
    a = physics_specs.radius

    def _get_vorticity_perturbation(lat, lon, sigma):
        del sigma  # unused.
        x = (np.sin(lat_location) * np.sin(lat) +
             np.cos(lat_location) * np.cos(lat) * np.cos(lon - lon_location))
        r = a * np.arccos(x)
        R = a * perturbation_radius  # pylint: disable=invalid-name
        return (u_p / a) * np.exp(-(r / R)**2) * (
            np.tan(lat) - (2 * ((a / R)**2) * np.arccos(x)) *
            (np.sin(lat_location) * np.cos(lat) -
             np.cos(lat_location) * np.sin(lat) * np.cos(lon - lon_location)) /
            (np.sqrt(1 - x**2)))

    def _get_divergence_perturbation(lat, lon, sigma):
        del sigma  # unused.
        x = (np.sin(lat_location) * np.sin(lat) +
             np.cos(lat_location) * np.cos(lat) * np.cos(lon - lon_location))
        r = a * np.arccos(x)
        R = a * perturbation_radius  # pylint: disable=invalid-name
        return (-2 * u_p * a / (R**2)) * np.exp(-(r / R)**2) * np.arccos(x) * (
            (np.cos(lat_location) * np.sin(lon - lon_location)) /
            (np.sqrt(1 - x**2)))

    lon, sin_lat = coords.horizontal.nodal_mesh
    lat = np.arcsin(sin_lat)
    nodal_vorticity = np.stack([
        _get_vorticity_perturbation(lat, lon, sigma)
        for sigma in coords.vertical.centers
    ])
    nodal_divergence = np.stack([
        _get_divergence_perturbation(lat, lon, sigma)
        for sigma in coords.vertical.centers
    ])
    modal_vorticity = coords.horizontal.to_modal(nodal_vorticity)
    modal_divergence = coords.horizontal.to_modal(nodal_divergence)
    state = primitive_equations.State(
        vorticity=modal_vorticity,
        divergence=modal_divergence,
        temperature_variation=np.zeros_like(modal_vorticity),
        log_surface_pressure=np.zeros_like(modal_vorticity[:1, ...]),
    )
    return state
