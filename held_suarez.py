import di
import functools
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import xarray

units = di.units


class HeldSuarezForcing:

    def __init__(
        self,
        coords,
        reference_temperature,
        p0=1e5 * units.pascal,
        sigma_b=0.7,
        kf=1 / (1 * units.day),
        ka=1 / (40 * units.day),
        ks=1 / (4 * units.day),
        minT=200 * units.degK,
        maxT=315 * units.degK,
        dTy=60 * units.degK,
        dThz=10 * units.degK,
    ):
        self.coords = coords
        self.reference_temperature = reference_temperature
        self.p0 = di.DEFAULT_SCALE.nondimensionalize(p0)
        self.sigma_b = sigma_b
        self.kf = di.DEFAULT_SCALE.nondimensionalize(kf)
        self.ka = di.DEFAULT_SCALE.nondimensionalize(ka)
        self.ks = di.DEFAULT_SCALE.nondimensionalize(ks)
        self.minT = di.DEFAULT_SCALE.nondimensionalize(minT)
        self.maxT = di.DEFAULT_SCALE.nondimensionalize(maxT)
        self.dTy = di.DEFAULT_SCALE.nondimensionalize(dTy)
        self.dThz = di.DEFAULT_SCALE.nondimensionalize(dThz)
        self.sigma = self.coords.vertical.centers
        _, sin_lat = self.coords.horizontal.nodal_mesh
        self.lat = np.arcsin(sin_lat)

    def kv(self):
        kv_coeff = self.kf * (np.maximum(0, (self.sigma - self.sigma_b) /
                                         (1 - self.sigma_b)))
        return kv_coeff[:, np.newaxis, np.newaxis]

    def kt(self):
        cutoff = np.maximum(0,
                            (self.sigma - self.sigma_b) / (1 - self.sigma_b))
        return self.ka + (self.ks - self.ka) * (
            cutoff[:, np.newaxis, np.newaxis] * np.cos(self.lat)**4)

    def equilibrium_temperature(self, nodal_surface_pressure):
        p_over_p0 = (self.sigma[:, np.newaxis, np.newaxis] *
                     nodal_surface_pressure / self.p0)
        temperature = p_over_p0**di.kappa * (
            self.maxT - self.dTy * np.sin(self.lat)**2 -
            self.dThz * jnp.log(p_over_p0) * np.cos(self.lat)**2)
        return jnp.maximum(self.minT, temperature)

    def explicit_terms(self, state):
        aux_state = di.compute_diagnostic_state(state=state,
                                                coords=self.coords)
        nodal_velocity_tendency = jax.tree.map(
            lambda x: -self.kv() * x / self.coords.horizontal.cos_lat**2,
            aux_state.cos_lat_u,
        )
        nodal_temperature = (
            self.reference_temperature[:, np.newaxis, np.newaxis] +
            aux_state.temperature_variation)
        nodal_log_surface_pressure = self.coords.horizontal.to_nodal(
            state.log_surface_pressure)
        nodal_surface_pressure = jnp.exp(nodal_log_surface_pressure)
        Teq = self.equilibrium_temperature(nodal_surface_pressure)
        nodal_temperature_tendency = -self.kt() * (nodal_temperature - Teq)
        temperature_tendency = self.coords.horizontal.to_modal(
            nodal_temperature_tendency)
        velocity_tendency = self.coords.horizontal.to_modal(
            nodal_velocity_tendency)
        vorticity_tendency = self.coords.horizontal.curl_cos_lat(
            velocity_tendency)
        divergence_tendency = self.coords.horizontal.div_cos_lat(
            velocity_tendency)
        log_surface_pressure_tendency = jnp.zeros_like(
            state.log_surface_pressure)
        return di.State(
            vorticity=vorticity_tendency,
            divergence=divergence_tendency,
            temperature_variation=temperature_tendency,
            log_surface_pressure=log_surface_pressure_tendency,
        )


def isothermal_rest_atmosphere(
    coords,
    tref=288.0 * units.degK,
    p0=1e5 * units.pascal,
    p1=0.0 * units.pascal,
):
    lon, sin_lat = coords.horizontal.nodal_mesh
    lat = np.arcsin(sin_lat)
    tref = di.DEFAULT_SCALE.nondimensionalize(units.Quantity(tref))
    p0 = di.DEFAULT_SCALE.nondimensionalize(units.Quantity(p0))
    p1 = di.DEFAULT_SCALE.nondimensionalize(units.Quantity(p1))
    orography = np.zeros_like(lat)

    def _get_vorticity(sigma, lon, lat):
        del sigma, lon
        return jnp.zeros_like(lat)

    def _get_surface_pressure(lon, lat, rng_key):

        def relative_pressure(altitude_m):
            g = 9.80665
            cp = 1004.68506
            T0 = 288.16
            M = 0.02896968
            R0 = 8.314462618
            return (1 - g * altitude_m / (cp * T0))**(cp * M / R0)

        altitude_m = di.DEFAULT_SCALE.dimensionalize(orography,
                                                     units.meter).magnitude
        surface_pressure = (p0 * np.ones(coords.surface_nodal_shape) *
                            relative_pressure(altitude_m))
        keys = jax.random.split(rng_key, 2)
        lon0 = jax.random.uniform(keys[1],
                                  minval=np.pi / 2,
                                  maxval=3 * np.pi / 2)
        lat0 = jax.random.uniform(keys[0], minval=-np.pi / 4, maxval=np.pi / 4)
        stddev = np.pi / 20
        k = 4
        perturbation = (jnp.exp(-((lon - lon0)**2) / (2 * stddev**2)) *
                        jnp.exp(-((lat - lat0)**2) /
                                (2 * stddev**2)) * jnp.sin(k * (lon - lon0)))
        return surface_pressure + p1 * perturbation

    def random_state_fn(rng_key: jnp.ndarray):
        nodal_vorticity = jnp.stack([
            _get_vorticity(sigma, lon, lat)
            for sigma in coords.vertical.centers
        ])
        modal_vorticity = coords.horizontal.to_modal(nodal_vorticity)
        nodal_surface_pressure = _get_surface_pressure(lon, lat, rng_key)
        return di.State(
            vorticity=modal_vorticity,
            divergence=jnp.zeros_like(modal_vorticity),
            temperature_variation=jnp.zeros_like(modal_vorticity),
            log_surface_pressure=(coords.horizontal.to_modal(
                jnp.log(nodal_surface_pressure))),
        )

    aux_features = {
        "orography": orography,
        "ref_temperatures": np.full((coords.vertical.layers, ), tref),
    }
    return random_state_fn, aux_features


layers = 24
coords = di.CoordinateSystem(
    horizontal=di.Grid(longitude_wavenumbers=22, total_wavenumbers=23, longitude_nodes=64, latitude_nodes=32),
    vertical=di.SigmaCoordinates.equidistant(layers),
)
p0 = 100e3 * units.pascal
p1 = 5e3 * units.pascal
rng_key = jax.random.PRNGKey(0)
initial_state_fn, aux_features = (isothermal_rest_atmosphere(coords=coords,
                                                             p0=p0,
                                                             p1=p1))
initial_state = initial_state_fn(rng_key)
ref_temps = aux_features["ref_temperatures"]
orography = di.truncated_modal_orography(aux_features["orography"], coords)
initial_state_dict = initial_state.asdict()
u, v = di.vor_div_to_uv_nodal(coords.horizontal, initial_state.vorticity,
                              initial_state.divergence)
initial_state_dict.update({"u": u, "v": v, "orography": orography})
dt_si = 5 * units.minute
save_every = 10 * units.minute
total_time = 24 * units.hour
inner_steps = int(save_every / dt_si)
outer_steps = int(total_time / save_every)
dt = di.DEFAULT_SCALE.nondimensionalize(dt_si)
primitive = di.PrimitiveEquations(ref_temps, orography, coords)
integrator = di.imex_rk_sil3
step_fn = integrator(primitive, dt)
filters = [di.exponential_step_filter(coords.horizontal, dt)]
step_fn = di.step_with_filters(step_fn, filters)
integrate_fn = jax.jit(
    di.trajectory_from_step(step_fn,
                            outer_steps=outer_steps,
                            inner_steps=inner_steps,
                            start_with_input=True))
times = save_every * np.arange(0, outer_steps)
final, trajectory = jax.block_until_ready(integrate_fn(initial_state))
hs = HeldSuarezForcing(coords=coords, reference_temperature=ref_temps, p0=p0)
dt_si = 10 * units.minute
save_every = 10 * units.day
total_time = 1200 * units.day
inner_steps = int(save_every / dt_si)
outer_steps = int(total_time / save_every)
dt = di.DEFAULT_SCALE.nondimensionalize(dt_si)
primitive = di.PrimitiveEquations(ref_temps, orography, coords)
hs_forcing = HeldSuarezForcing(coords=coords,
                               reference_temperature=ref_temps,
                               p0=p0)
primitive_with_hs = di.compose_equations([primitive, hs_forcing])
step_fn = di.imex_rk_sil3(primitive_with_hs, dt)
filters = [
    di.exponential_step_filter(coords.horizontal,
                               dt,
                               tau=0.0087504,
                               order=1.5,
                               cutoff=0.8),
]
step_fn = di.step_with_filters(step_fn, filters)
integrate_fn = jax.jit(
    di.trajectory_from_step(step_fn,
                            outer_steps=outer_steps,
                            inner_steps=inner_steps))
times = save_every * np.arange(1, outer_steps + 1)
final, trajectory = jax.block_until_ready(integrate_fn(initial_state))
start_time = 200
dt_si = 10 * units.minute
save_every = 6 * units.hours
total_time = 1 * units.week
inner_steps = int(save_every / dt_si)
outer_steps = int(total_time / save_every)
dt = di.DEFAULT_SCALE.nondimensionalize(dt_si)
step_fn = di.imex_rk_sil3(primitive_with_hs, dt)
filters = [
    di.exponential_step_filter(coords.horizontal,
                               dt,
                               tau=0.0087504,
                               order=1.5,
                               cutoff=0.8),
]
step_fn = di.step_with_filters(step_fn, filters)
integrate_fn = jax.jit(
    di.trajectory_from_step(
        step_fn,
        outer_steps=outer_steps,
        inner_steps=inner_steps,
    ))
times = save_every * np.arange(1, outer_steps + 1)
final, trajectory = jax.block_until_ready(integrate_fn(final))
trajectory_dict = trajectory.asdict()
u, v = di.vor_div_to_uv_nodal(coords.horizontal, trajectory.vorticity,
                              trajectory.divergence)
trajectory_dict.update({"u": u, "v": v})
f0 = di.maybe_to_nodal(trajectory_dict, coords=coords)
plt.contourf(f0["temperature_variation"][-1, 22, :, :])
plt.savefig("h.12.png")
np.asarray(f0["temperature_variation"]).tofile("h.12.raw")
plt.close()
