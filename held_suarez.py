import di
import functools
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import xarray

units = di.units


def dimensionalize(x, unit):
    dimensionalize = functools.partial(di.DEFAULT_SCALE.dimensionalize,
                                       unit=unit)
    return xarray.apply_ufunc(dimensionalize, x)


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


def trajectory_to_xarray(coords, trajectory, times):
    trajectory_dict, _ = di.as_dict(trajectory)
    u, v = di.vor_div_to_uv_nodal(coords.horizontal, trajectory.vorticity,
                                  trajectory.divergence)
    trajectory_dict.update({"u": u, "v": v})
    nodal_trajectory_fields = di.maybe_to_nodal(trajectory_dict, coords=coords)
    trajectory_ds = di.data_to_xarray(nodal_trajectory_fields,
                                      coords=coords,
                                      times=times)
    trajectory_ds["surface_pressure"] = np.exp(
        trajectory_ds.log_surface_pressure[:, 0, :, :])
    temperature = di.temperature_variation_to_absolute(
        trajectory_ds.temperature_variation.data, ref_temps)
    trajectory_ds = trajectory_ds.assign(
        temperature=(trajectory_ds.temperature_variation.dims, temperature))
    total_layer_ke = coords.horizontal.integrate(u**2 + v**2)
    total_ke_cumulative = di.cumulative_sigma_integral(total_layer_ke,
                                                       coords.vertical,
                                                       axis=-1)
    total_ke = total_ke_cumulative[..., -1]
    trajectory_ds = trajectory_ds.assign(total_kinetic_energy=(("time"),
                                                               total_ke))
    return trajectory_ds


def ds_held_suarez_forcing(coords):
    grid = coords.horizontal
    sigma = coords.vertical.centers
    lon, _ = grid.nodal_mesh
    surface_pressure = di.DEFAULT_SCALE.nondimensionalize(p0) * np.ones_like(
        lon)
    dims = ("sigma", "lon", "lat")
    return xarray.Dataset(
        data_vars={
            "surface_pressure": (("lon", "lat"), surface_pressure),
            "eq_temp": (dims, hs.equilibrium_temperature(surface_pressure)),
            "kt": (dims, hs.kt()),
            "kv": ("sigma", hs.kv()[:, 0, 0]),
        },
        coords={
            "lon": grid.nodal_axes[0] * 180 / np.pi,
            "lat": np.arcsin(grid.nodal_axes[1]) * 180 / np.pi,
            "sigma": sigma,
        },
    )


def linspace_step(start, stop, step):
    num = round((stop - start) / step) + 1
    return np.linspace(start, stop, num)


layers = 24
coords = di.CoordinateSystem(
    horizontal=di.Grid.T42(),
    vertical=di.SigmaCoordinates.equidistant(layers),
)
p0 = 100e3 * units.pascal
p1 = 5e3 * units.pascal
rng_key = jax.random.PRNGKey(0)
initial_state_fn, aux_features = (di.isothermal_rest_atmosphere(coords=coords,
                                                                p0=p0,
                                                                p1=p1))
initial_state = initial_state_fn(rng_key)
ref_temps = aux_features["ref_temperatures"]
orography = di.truncated_modal_orography(aux_features["orography"], coords)
initial_state_dict, _ = di.as_dict(initial_state)
u, v = di.vor_div_to_uv_nodal(coords.horizontal, initial_state.vorticity,
                              initial_state.divergence)
initial_state_dict.update({"u": u, "v": v, "orography": orography})
nodal_steady_state_fields = di.maybe_to_nodal(initial_state_dict,
                                              coords=coords)
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
temperature = dimensionalize(ds["eq_temp"], units.degK)
surface_pressure = dimensionalize(ds["surface_pressure"], units.pascal)
pressure = ds.sigma * surface_pressure
kappa = di.KAPPA
potential_temperature = temperature * (pressure / p0)**-kappa
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
ds = trajectory_to_xarray(coords, jax.device_get(trajectory), times)
start_time = 200
mask = ds["time"] > start_time
data_array = ds["temperature"]
data_array = dimensionalize(data_array, units.degK)
levels = linspace_step(190, 305, 5)
data_array.isel(time=mask).mean(["lon", "time"]).plot.contour(x="lat",
                                                              y="level",
                                                              levels=levels,
                                                              size=5,
                                                              aspect=1.5)
ax = plt.gca()
ax.set_ylim((1, 0))
mask = ds["time"] > start_time
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
final_2, trajectory_2 = jax.block_until_ready(integrate_fn(final))
ds = trajectory_to_xarray(coords, trajectory_2, times)
data_array = ds["temperature"]
data_array.thin(time=4).isel(level=-10).plot(x="lon", y="lat", col="time")
plt.savefig("h.12.png")
plt.close()
