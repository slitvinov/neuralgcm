from __future__ import annotations
import dataclasses
import functools
from typing import Any, Callable, Mapping, Sequence, Union
from dinosaur import coordinate_systems
from dinosaur import jax_numpy_utils
from dinosaur import scales
from dinosaur import sigma_coordinates
from dinosaur import spherical_harmonic
from dinosaur import time_integration
from dinosaur import typing
from dinosaur import vertical_interpolation
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
import tree_math

units = scales.units
Array = typing.Array
Numeric = typing.Numeric
Quantity = typing.Quantity
OrographyInitFn = Callable[..., Array]
einsum = functools.partial(jnp.einsum, precision=jax.lax.Precision.HIGHEST)


@tree_math.struct
class State:
    vorticity: Array
    divergence: Array
    temperature_variation: Array
    log_surface_pressure: Array
    tracers: Mapping[str, Array] = dataclasses.field(default_factory=dict)
    sim_time: Union[float, None] = None


def _asdict(state: State) -> dict[str, Any]:
    return {
        field.name: getattr(state, field.name)
        for field in state.fields
        if field.name != 'sim_time' or state.sim_time is not None
    }


State.asdict = _asdict
StateWithTime = State  # deprecated alias


class StateShapeError(Exception):

    def validate_state_shape(state: State,
                             coords: coordinate_systems.CoordinateSystem):
        if state.vorticity.shape != coords.modal_shape:
            raise StateShapeError(
                f'Expected vorticity shape {coords.modal_shape}; '
                f'got shape {state.vorticity.shape}.')
        if state.divergence.shape != coords.modal_shape:
            raise StateShapeError(
                f'Expected divergence shape {coords.modal_shape}; '
                f'got shape {state.divergence.shape}.')
        if state.temperature_variation.shape != coords.modal_shape:
            raise StateShapeError(
                f'Expected temperature_variation shape {coords.modal_shape}; '
                f'got shape {state.temperature_variation.shape}.')
        if state.log_surface_pressure.shape != coords.surface_modal_shape:
            raise StateShapeError(
                f'Expected log_surface_pressure shape {coords.surface_modal_shape}; '
                f'got shape {state.log_surface_pressure.shape}.')
        for tracer_name, array in state.tracers.items():
            if array.shape[-3:] != coords.modal_shape:
                raise StateShapeError(
                    f'Expected tracer {tracer_name} shape {coords.modal_shape}; '
                    f'got shape {array.shape}.')


@tree_math.struct
class DiagnosticState:
    vorticity: Array
    divergence: Array
    temperature_variation: Array
    cos_lat_u: tuple[Array, Array]
    sigma_dot_explicit: Array
    sigma_dot_full: Array
    cos_lat_grad_log_sp: Array
    u_dot_grad_log_sp: Array
    tracers: Mapping[str, Array]


@jax.named_call
def compute_diagnostic_state(
    state: State,
    coords: coordinate_systems.CoordinateSystem,
) -> DiagnosticState:

    def to_nodal_fn(x):
        return coords.horizontal.to_nodal(x)

    nodal_vorticity = to_nodal_fn(state.vorticity)
    nodal_divergence = to_nodal_fn(state.divergence)
    nodal_temperature_variation = to_nodal_fn(state.temperature_variation)
    tracers = to_nodal_fn(state.tracers)
    nodal_cos_lat_u = jax.tree_util.tree_map(
        to_nodal_fn,
        spherical_harmonic.get_cos_lat_vector(state.vorticity,
                                              state.divergence,
                                              coords.horizontal,
                                              clip=False),
    )
    cos_lat_grad_log_sp = coords.horizontal.cos_lat_grad(
        state.log_surface_pressure, clip=False)
    nodal_cos_lat_grad_log_sp = to_nodal_fn(cos_lat_grad_log_sp)
    nodal_u_dot_grad_log_sp = sum(
        jax.tree_util.tree_map(
            lambda x, y: x * y * coords.horizontal.sec2_lat,
            nodal_cos_lat_u,
            nodal_cos_lat_grad_log_sp,
        ))
    f_explicit = sigma_coordinates.cumulative_sigma_integral(
        nodal_u_dot_grad_log_sp, coords.vertical)
    f_full = sigma_coordinates.cumulative_sigma_integral(
        nodal_divergence + nodal_u_dot_grad_log_sp, coords.vertical)
    sum_𝜎 = np.cumsum(coords.vertical.layer_thickness)[:, np.newaxis,
                                                       np.newaxis]
    sigma_dot_explicit = lax.slice_in_dim(
        sum_𝜎 * lax.slice_in_dim(f_explicit, -1, None) - f_explicit, 0, -1)
    sigma_dot_full = lax.slice_in_dim(
        sum_𝜎 * lax.slice_in_dim(f_full, -1, None) - f_full, 0, -1)
    return DiagnosticState(
        vorticity=nodal_vorticity,
        divergence=nodal_divergence,
        temperature_variation=nodal_temperature_variation,
        cos_lat_u=nodal_cos_lat_u,
        sigma_dot_explicit=sigma_dot_explicit,
        sigma_dot_full=sigma_dot_full,
        cos_lat_grad_log_sp=nodal_cos_lat_grad_log_sp,
        u_dot_grad_log_sp=nodal_u_dot_grad_log_sp,
        tracers=tracers,
    )


def compute_vertical_velocity(
        state: State,
        coords: coordinate_systems.CoordinateSystem) -> jax.Array:
    sigma_dot_boundaries = compute_diagnostic_state(state,
                                                    coords).sigma_dot_full
    assert sigma_dot_boundaries.ndim == 3
    sigma_dot_padded = jnp.pad(sigma_dot_boundaries, [(1, 1), (0, 0), (0, 0)])
    return 0.5 * (sigma_dot_padded[1:] + sigma_dot_padded[:-1])


@dataclasses.dataclass(frozen=True)
class PrimitiveEquationsSpecs:
    radius: float
    angular_velocity: float
    gravity_acceleration: float
    ideal_gas_constant: float
    water_vapor_gas_constant: float
    water_vapor_isobaric_heat_capacity: float
    kappa: float
    scale: scales.ScaleProtocol

    @property
    def R(self) -> float:
        return self.ideal_gas_constant

    @property
    def R_vapor(self) -> float:
        return self.water_vapor_gas_constant

    @property
    def g(self) -> float:
        return self.gravity_acceleration

    @property
    def Cp(self) -> float:
        return self.ideal_gas_constant / self.kappa

    @property
    def Cp_vapor(self) -> float:
        return self.water_vapor_isobaric_heat_capacity

    def nondimensionalize(self, quantity: Quantity) -> Numeric:
        return self.scale.nondimensionalize(quantity)

    def nondimensionalize_timedelta64(self,
                                      timedelta: np.timedelta64) -> Numeric:
        base_unit = 's'
        return self.scale.nondimensionalize(
            timedelta / np.timedelta64(1, base_unit) * units(base_unit))

    def dimensionalize(self, value: Numeric, unit: units.Unit) -> Quantity:
        return self.scale.dimensionalize(value, unit)

    def dimensionalize_timedelta64(self, value: Numeric) -> np.timedelta64:
        base_unit = 's'  # return value is rounded down to nearest base_unit
        dt = self.scale.dimensionalize(value, units(base_unit)).m
        if isinstance(dt, np.ndarray):
            return dt.astype(f'timedelta64[{base_unit}]')
        else:
            return np.timedelta64(int(dt), base_unit)

    @classmethod
    def from_si(
        cls,
        radius_si: Quantity = scales.RADIUS,
        angular_velocity_si: Quantity = scales.ANGULAR_VELOCITY,
        gravity_acceleration_si: Quantity = scales.GRAVITY_ACCELERATION,
        ideal_gas_constant_si: Quantity = scales.IDEAL_GAS_CONSTANT,
        water_vapor_gas_constant_si: Quantity = scales.IDEAL_GAS_CONSTANT_H20,
        water_vapor_isobaric_heat_capacity_si: Quantity = scales.
        WATER_VAPOR_CP,
        kappa_si: Quantity = scales.KAPPA,
        scale: scales.ScaleProtocol = scales.DEFAULT_SCALE,
    ) -> PrimitiveEquationsSpecs:
        return cls(
            scale.nondimensionalize(radius_si),
            scale.nondimensionalize(angular_velocity_si),
            scale.nondimensionalize(gravity_acceleration_si),
            scale.nondimensionalize(ideal_gas_constant_si),
            scale.nondimensionalize(water_vapor_gas_constant_si),
            scale.nondimensionalize(water_vapor_isobaric_heat_capacity_si),
            scale.nondimensionalize(kappa_si),
            scale,
        )


def get_sigma_ratios(
    coordinates: sigma_coordinates.SigmaCoordinates, ) -> np.ndarray:
    alpha = np.diff(np.log(coordinates.centers), append=0) / 2
    alpha[-1] = -np.log(coordinates.centers[-1])
    return alpha


def get_geopotential_weights(
    coordinates: sigma_coordinates.SigmaCoordinates,
    ideal_gas_constant: float,
) -> np.ndarray:
    alpha = get_sigma_ratios(coordinates)
    weights = np.zeros([coordinates.layers, coordinates.layers])
    for j in range(coordinates.layers):
        weights[j, j] = alpha[j]
        for k in range(j + 1, coordinates.layers):
            weights[j, k] = alpha[k] + alpha[k - 1]
    return ideal_gas_constant * weights


def get_geopotential_diff(
    temperature: Array,
    coordinates: sigma_coordinates.SigmaCoordinates,
    ideal_gas_constant: float,
    method: str = 'dense',
    sharding: Union[jax.sharding.NamedSharding, None] = None,
) -> jax.Array:
    if method == 'dense':
        weights = get_geopotential_weights(coordinates, ideal_gas_constant)
        return _vertical_matvec(weights, temperature)
    elif method == 'sparse':
        assert False
    else:
        raise ValueError(f'unknown {method=} for get_geopotential_diff')


def get_geopotential(
    temperature_variation: Array,
    reference_temperature: Array,
    orography: Array,
    coordinates: sigma_coordinates.SigmaCoordinates,
    gravity_acceleration: float,
    ideal_gas_constant: float,
    sharding: Union[jax.sharding.NamedSharding, None] = None,
) -> jnp.ndarray:
    surface_geopotential = orography * gravity_acceleration
    temperature = spherical_harmonic.add_constant(temperature_variation,
                                                  reference_temperature)
    geopotential_diff = get_geopotential_diff(temperature,
                                              coordinates,
                                              ideal_gas_constant,
                                              sharding=sharding)
    return surface_geopotential + geopotential_diff


def get_geopotential_with_moisture(
    temperature: typing.Array,
    specific_humidity: typing.Array,
    nodal_orography: typing.Array,
    coordinates: sigma_coordinates.SigmaCoordinates,
    gravity_acceleration: float,
    ideal_gas_constant: float,
    water_vapor_gas_constant: float,
    sharding: Union[jax.sharding.NamedSharding, None] = None,
    clouds: Union[typing.Array, None] = None,
) -> jnp.ndarray:
    gas_const_ratio = water_vapor_gas_constant / ideal_gas_constant
    surface_geopotential = nodal_orography * gravity_acceleration
    if clouds is None:
        clouds = 0.0
    virtual_temp = temperature * (1 +
                                  (gas_const_ratio - 1) * specific_humidity -
                                  clouds)
    geopotential_diff = get_geopotential_diff(virtual_temp,
                                              coordinates,
                                              ideal_gas_constant,
                                              sharding=sharding)
    return surface_geopotential + geopotential_diff


def get_temperature_implicit_weights(
    coordinates: sigma_coordinates.SigmaCoordinates,
    reference_temperature: np.ndarray,
    kappa: float,
) -> np.ndarray:
    if (reference_temperature.ndim != 1
            or reference_temperature.shape[-1] != coordinates.layers):
        raise ValueError(
            '`reference_temp` must be a vector of length `coordinates.layers`; '
            f'got shape {reference_temperature.shape} and '
            f'{coordinates.layers} layers.')
    p = np.tril(np.ones([coordinates.layers, coordinates.layers]))
    alpha = get_sigma_ratios(coordinates)[..., np.newaxis]
    p_alpha = p * alpha
    p_alpha_shifted = np.roll(p_alpha, 1, axis=0)
    p_alpha_shifted[0] = 0
    h0 = (kappa * reference_temperature[..., np.newaxis] *
          (p_alpha + p_alpha_shifted) /
          coordinates.layer_thickness[..., np.newaxis])
    temp_diff = np.diff(reference_temperature)
    thickness_sum = (coordinates.layer_thickness[:-1] +
                     coordinates.layer_thickness[1:])
    k0 = np.concatenate((temp_diff / thickness_sum, [0]), axis=0)[...,
                                                                  np.newaxis]
    thickness_cumulative = np.cumsum(coordinates.layer_thickness)[...,
                                                                  np.newaxis]
    k1 = p - thickness_cumulative
    k = k0 * k1
    k_shifted = np.roll(k, 1, axis=0)
    k_shifted[0] = 0
    return (h0 - k - k_shifted) * coordinates.layer_thickness


def get_temperature_implicit(
    divergence: Array,
    coordinates: sigma_coordinates.SigmaCoordinates,
    reference_temperature: np.ndarray,
    kappa: float,
    method: str = 'dense',
    sharding: Union[jax.sharding.NamedSharding, None] = None,
) -> jax.Array:
    weights = -get_temperature_implicit_weights(coordinates,
                                                reference_temperature, kappa)
    if method == 'dense':
        return _vertical_matvec(weights, divergence)
    elif method == 'sparse':
        assert False
    else:
        raise ValueError(f'unknown {method=} for get_temperature_implicit')


@jax.named_call
def _vertical_matvec(a: Array, x: Array) -> jax.Array:
    return einsum('gh,...hml->...gml', a, x)


@jax.named_call
def _vertical_matvec_per_wavenumber(a: Array, x: Array) -> jax.Array:
    return einsum('lgh,...hml->...gml', a, x)


def _get_implicit_term_matrix(eta, coords, reference_temperature, kappa,
                              ideal_gas_constant) -> np.ndarray:
    eye = np.eye(coords.vertical.layers)[np.newaxis]
    lam = coords.horizontal.laplacian_eigenvalues
    g = get_geopotential_weights(coords.vertical, ideal_gas_constant)
    r = ideal_gas_constant
    h = get_temperature_implicit_weights(coords.vertical,
                                         reference_temperature, kappa)
    t = reference_temperature[:, np.newaxis]
    thickness = coords.vertical.layer_thickness[np.newaxis, np.newaxis, :]
    l = coords.horizontal.modal_shape[1]
    j = k = coords.vertical.layers
    row0 = np.concatenate(
        [
            np.broadcast_to(eye, [l, j, k]),
            eta * np.einsum('l,jk->ljk', lam, g),
            eta * r * np.einsum('l,jo->ljo', lam, t),
        ],
        axis=2,
    )
    row1 = np.concatenate(
        [
            eta * np.broadcast_to(h[np.newaxis], [l, j, k]),
            np.broadcast_to(eye, [l, j, k]),
            np.zeros([l, j, 1]),
        ],
        axis=2,
    )
    row2 = np.concatenate(
        [
            np.broadcast_to(eta * thickness, [l, 1, k]),
            np.zeros([l, 1, k]),
            np.ones([l, 1, 1]),
        ],
        axis=2,
    )
    return np.concatenate((row0, row1, row2), axis=1)


def div_sec_lat(m_component: Array, n_component: Array,
                grid: spherical_harmonic.Grid) -> Array:
    m_component = grid.to_modal(m_component * grid.sec2_lat)
    n_component = grid.to_modal(n_component * grid.sec2_lat)
    return grid.div_cos_lat((m_component, n_component), clip=False)


def truncated_modal_orography(
    orography: Array,
    coords: coordinate_systems.CoordinateSystem,
    wavenumbers_to_clip: int = 1,
) -> Array:
    grid = coords.horizontal
    expected_shape = grid.nodal_shape
    if orography.shape != expected_shape:
        raise ValueError(
            f'Expected nodal orography with shape={expected_shape}')
    return grid.clip_wavenumbers(grid.to_modal(orography),
                                 n=wavenumbers_to_clip)


def filtered_modal_orography(
        orography: Array,
        coords: coordinate_systems.CoordinateSystem,
        input_coords: Union[coordinate_systems.CoordinateSystem, None] = None,
        filter_fns: Sequence[typing.PostProcessFn] = tuple(),
) -> Array:
    if input_coords is None:
        input_coords = coords
    expected_shape = input_coords.horizontal.nodal_shape
    if orography.shape != expected_shape:
        raise ValueError(
            f'Expected nodal orography with shape={expected_shape}')
    interpolate_fn = coordinate_systems.get_spectral_interpolate_fn(
        input_coords, coords, expect_same_vertical=False)
    modal_orography = interpolate_fn(
        input_coords.horizontal.to_modal(orography))
    for filter_fn in filter_fns:
        modal_orography = filter_fn(modal_orography)
    return modal_orography


@dataclasses.dataclass
class PrimitiveEquations(time_integration.ImplicitExplicitODE):
    reference_temperature: np.ndarray
    orography: Array
    coords: coordinate_systems.CoordinateSystem
    physics_specs: PrimitiveEquationsSpecs
    vertical_matmul_method: Union[str, None] = dataclasses.field(default=None)
    implicit_inverse_method: str = dataclasses.field(default='split')
    vertical_advection: Callable[..., jax.Array] = dataclasses.field(
        default=sigma_coordinates.centered_vertical_advection)
    include_vertical_advection: bool = dataclasses.field(default=True)

    def __post_init__(self):
        if not np.allclose(self.coords.horizontal.radius,
                           self.physics_specs.radius,
                           rtol=1e-5):
            raise ValueError(
                'inconsistent radius between coordinates and constants: '
                f'{self.coords.horizontal.radius=} != {self.physics_specs.radius=}'
            )

    @property
    def coriolis_parameter(self) -> Array:
        _, sin_lat = self.coords.horizontal.nodal_mesh
        return 2 * self.physics_specs.angular_velocity * sin_lat

    @property
    def T_ref(self) -> Array:
        return self.reference_temperature[..., np.newaxis, np.newaxis]

    @jax.named_call
    def _vertical_tendency(self, w: Array, x: Array) -> Array:
        return self.vertical_advection(w, x, self.coords.vertical)

    @jax.named_call
    def _t_omega_over_sigma_sp(self, temperature_field: Array, g_term: Array,
                               v_dot_grad_log_sp: Array) -> Array:
        f = sigma_coordinates.cumulative_sigma_integral(
            g_term, self.coords.vertical, sharding=self.coords.dycore_sharding)
        alpha = get_sigma_ratios(self.coords.vertical)
        alpha = alpha[:, np.newaxis,
                      np.newaxis]  # make alpha broadcast to `f`.
        del_𝜎 = self.coords.vertical.layer_thickness[:, np.newaxis, np.newaxis]
        padding = [(1, 0), (0, 0), (0, 0)]
        g_part = (alpha * f + jnp.pad(alpha * f, padding)[:-1, ...]) / del_𝜎
        return temperature_field * (v_dot_grad_log_sp - g_part)

    @jax.named_call
    def kinetic_energy_tendency(self, aux_state: DiagnosticState) -> Array:
        nodal_cos_lat_u2 = jnp.stack(aux_state.cos_lat_u)**2
        kinetic = nodal_cos_lat_u2.sum(0) * self.coords.horizontal.sec2_lat / 2
        return -self.coords.horizontal.laplacian(
            self.coords.horizontal.to_modal(kinetic))

    @jax.named_call
    def orography_tendency(self) -> Array:
        return -self.physics_specs.g * self.coords.horizontal.laplacian(
            self.orography)

    @jax.named_call
    def curl_and_div_tendencies(
        self,
        aux_state: DiagnosticState,
    ) -> tuple[Array, Array]:
        sec2_lat = self.coords.horizontal.sec2_lat
        u, v = aux_state.cos_lat_u
        total_vorticity = aux_state.vorticity + self.coriolis_parameter
        nodal_vorticity_u = -v * total_vorticity * sec2_lat
        nodal_vorticity_v = u * total_vorticity * sec2_lat
        d𝜎_dt = aux_state.sigma_dot_full
        if self.include_vertical_advection:
            sigma_dot_u = -self._vertical_tendency(d𝜎_dt, u)
            sigma_dot_v = -self._vertical_tendency(d𝜎_dt, v)
        else:
            sigma_dot_u = 0
            sigma_dot_v = 0
        rt = self.physics_specs.R * aux_state.temperature_variation
        grad_log_ps_u, grad_log_ps_v = aux_state.cos_lat_grad_log_sp
        vertical_term_u = (sigma_dot_u + rt * grad_log_ps_u) * sec2_lat
        vertical_term_v = (sigma_dot_v + rt * grad_log_ps_v) * sec2_lat
        combined_u = self.coords.horizontal.to_modal(nodal_vorticity_u +
                                                     vertical_term_u)
        combined_v = self.coords.horizontal.to_modal(nodal_vorticity_v +
                                                     vertical_term_v)
        dζ_dt = -self.coords.horizontal.curl_cos_lat(
            (combined_u, combined_v), clip=False)
        d𝛅_dt = -self.coords.horizontal.div_cos_lat(
            (combined_u, combined_v), clip=False)
        return (dζ_dt, d𝛅_dt)

    @jax.named_call
    def nodal_temperature_vertical_tendency(
        self,
        aux_state: DiagnosticState,
    ) -> Union[Array, float]:
        sigma_dot_explicit = aux_state.sigma_dot_explicit
        sigma_dot_full = aux_state.sigma_dot_full
        temperature_variation = aux_state.temperature_variation
        if self.include_vertical_advection:
            tendency = self._vertical_tendency(sigma_dot_full,
                                               temperature_variation)
        else:
            tendency = 0
        if np.unique(self.T_ref.ravel()).size > 1:
            tendency += self._vertical_tendency(sigma_dot_explicit, self.T_ref)
        return tendency

    @jax.named_call
    def horizontal_scalar_advection(
        self,
        scalar: Array,
        aux_state: DiagnosticState,
    ) -> tuple[Array, Array]:
        u, v = aux_state.cos_lat_u
        nodal_terms = scalar * aux_state.divergence
        modal_terms = -div_sec_lat(u * scalar, v * scalar,
                                   self.coords.horizontal)
        return nodal_terms, modal_terms

    @jax.named_call
    def nodal_temperature_adiabatic_tendency(
            self, aux_state: DiagnosticState) -> Array:
        g_explicit = aux_state.u_dot_grad_log_sp
        g_full = g_explicit + aux_state.divergence
        mean_t_part = self._t_omega_over_sigma_sp(self.T_ref, g_explicit,
                                                  aux_state.u_dot_grad_log_sp)
        variation_t_part = self._t_omega_over_sigma_sp(
            aux_state.temperature_variation, g_full,
            aux_state.u_dot_grad_log_sp)
        return self.physics_specs.kappa * (mean_t_part + variation_t_part)

    @jax.named_call
    def nodal_log_pressure_tendency(self, aux_state: DiagnosticState) -> Array:
        g = aux_state.u_dot_grad_log_sp
        return -sigma_coordinates.sigma_integral(g, self.coords.vertical)

    @jax.named_call
    def explicit_terms(self, state: State) -> State:
        aux_state = compute_diagnostic_state(state, self.coords)
        vorticity_tendency, divergence_dot = self.curl_and_div_tendencies(
            aux_state)
        kinetic_energy_tendency = self.kinetic_energy_tendency(aux_state)
        orography_tendency = self.orography_tendency()
        horizontal_tendency_fn = functools.partial(
            self.horizontal_scalar_advection, aux_state=aux_state)
        dT_dt_horizontal_nodal, dT_dt_horizontal_modal = horizontal_tendency_fn(
            aux_state.temperature_variation)
        tracers_horizontal_nodal_and_modal = jax.tree_util.tree_map(
            horizontal_tendency_fn, aux_state.tracers)
        dT_dt_vertical = self.nodal_temperature_vertical_tendency(aux_state)
        dT_dt_adiabatic = self.nodal_temperature_adiabatic_tendency(aux_state)
        log_sp_tendency = self.nodal_log_pressure_tendency(aux_state)
        sigma_dot_full = aux_state.sigma_dot_full
        if self.include_vertical_advection:
            vertical_tendency_fn = functools.partial(self._vertical_tendency,
                                                     sigma_dot_full)
        else:
            vertical_tendency_fn = lambda x: 0
        tracers_vertical_nodal = jax.tree_util.tree_map(
            vertical_tendency_fn, aux_state.tracers)
        to_modal_fn = self.coords.horizontal.to_modal
        divergence_tendency = (divergence_dot + kinetic_energy_tendency +
                               orography_tendency)
        temperature_tendency = (to_modal_fn(dT_dt_horizontal_nodal +
                                            dT_dt_vertical + dT_dt_adiabatic) +
                                dT_dt_horizontal_modal)
        log_surface_pressure_tendency = to_modal_fn(log_sp_tendency)
        tracers_tendency = jax.tree_util.tree_map(
            lambda x, y_z: to_modal_fn(x + y_z[0]) + y_z[1],
            tracers_vertical_nodal,
            tracers_horizontal_nodal_and_modal,
        )
        tendency = State(
            vorticity=vorticity_tendency,
            divergence=divergence_tendency,
            temperature_variation=temperature_tendency,
            log_surface_pressure=log_surface_pressure_tendency,
            tracers=tracers_tendency,
            sim_time=None if state.sim_time is None else 1.0,
        )
        return self.coords.horizontal.clip_wavenumbers(tendency)

    @jax.named_call
    def implicit_terms(self, state: State) -> State:
        method = self.vertical_matmul_method
        if method is None:
            mesh = self.coords.spmd_mesh
            method = 'sparse' if mesh is not None and mesh.shape[
                'z'] > 1 else 'dense'
        geopotential_diff = get_geopotential_diff(
            state.temperature_variation,
            self.coords.vertical,
            self.physics_specs.R,
            method=method,
            sharding=self.coords.dycore_sharding,
        )
        rt_log_p = (self.physics_specs.ideal_gas_constant * self.T_ref *
                    state.log_surface_pressure)
        vorticity_implicit = jnp.zeros_like(state.vorticity)
        divergence_implicit = -self.coords.horizontal.laplacian(
            geopotential_diff + rt_log_p)
        temperature_variation_implicit = get_temperature_implicit(
            state.divergence,
            self.coords.vertical,
            self.reference_temperature,
            self.physics_specs.kappa,
            method=method,
            sharding=self.coords.dycore_sharding,
        )
        log_surface_pressure_implicit = -_vertical_matvec(
            self.coords.vertical.layer_thickness[np.newaxis], state.divergence)
        tracers_implicit = jax.tree_util.tree_map(jnp.zeros_like,
                                                  state.tracers)
        return State(
            vorticity=vorticity_implicit,
            divergence=divergence_implicit,
            temperature_variation=temperature_variation_implicit,
            log_surface_pressure=log_surface_pressure_implicit,
            tracers=tracers_implicit,
            sim_time=None if state.sim_time is None else 0.0,
        )

    @jax.named_call
    def implicit_inverse(self, state: State, step_size: float) -> State:
        if isinstance(step_size, jax.core.Tracer):
            raise TypeError(
                f'`step_size` must be concrete but a Tracer was passed: {step_size}. '
                'This error is likely caused by '
                '`jax.jit(primitive.inverse_terms)(state, eta). Instead, do '
                '`jax.jit(lambda s: primitive.inverse_terms(s, eta=eta))(state)`.'
            )
        implicit_matrix = _get_implicit_term_matrix(
            step_size,
            self.coords,
            self.reference_temperature,
            self.physics_specs.kappa,
            self.physics_specs.R,
        )
        assert implicit_matrix.dtype == np.float64
        layers = self.coords.vertical.layers
        div = slice(0, layers)
        temp = slice(layers, 2 * layers)
        logp = slice(2 * layers, 2 * layers + 1)
        temp_logp = slice(layers, 2 * layers + 1)

        def named_vertical_matvec(name):
            return jax.named_call(_vertical_matvec_per_wavenumber, name=name)

        if self.implicit_inverse_method == 'split':
            inverse = np.linalg.inv(implicit_matrix)
            assert not np.isnan(inverse).any()
            inverted_divergence = (
                named_vertical_matvec('div_from_div')(inverse[:, div, div],
                                                      state.divergence) +
                named_vertical_matvec('div_from_temp')(
                    inverse[:, div, temp], state.temperature_variation) +
                named_vertical_matvec('div_from_logp')(
                    inverse[:, div, logp], state.log_surface_pressure))
            inverted_temperature_variation = (
                named_vertical_matvec('temp_from_div')(inverse[:, temp, div],
                                                       state.divergence) +
                named_vertical_matvec('temp_from_temp')(
                    inverse[:, temp, temp], state.temperature_variation) +
                named_vertical_matvec('temp_from_logp')(
                    inverse[:, temp, logp], state.log_surface_pressure))
            inverted_log_surface_pressure = (
                named_vertical_matvec('logp_from_div')(inverse[:, logp, div],
                                                       state.divergence) +
                named_vertical_matvec('logp_from_temp')(
                    inverse[:, logp, temp], state.temperature_variation) +
                named_vertical_matvec('logp_from_logp')(
                    inverse[:, logp, logp], state.log_surface_pressure))
        else:
            raise ValueError(
                f'invalid implicit_inverse_method {self.implicit_inverse_method}'
            )
        inverted_vorticity = state.vorticity
        inverted_tracers = state.tracers
        return State(
            inverted_vorticity,
            inverted_divergence,
            inverted_temperature_variation,
            inverted_log_surface_pressure,
            inverted_tracers,
            sim_time=state.sim_time,
        )


PrimitiveEquationsWithTime = PrimitiveEquations  # deprecated alias
