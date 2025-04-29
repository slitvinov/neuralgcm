import functools
import jax
import numpy as np
import matplotlib.pyplot as plt
import xarray
import di

units = di.units


def dimensionalize(x, unit):
    """Dimensionalizes `xarray.DataArray`s."""
    dimensionalize = functools.partial(di.DEFAULT_SCALE.dimensionalize, unit=unit)
    return xarray.apply_ufunc(dimensionalize, x)


units = di.units
layers = 24
coords = di.CoordinateSystem(
    horizontal=di.Grid.T42(),
    vertical=di.SigmaCoordinates.equidistant(layers),
)
physics_specs = di.PrimitiveEquationsSpecs.from_si()
p0 = 100e3 * units.pascal
p1 = 5e3 * units.pascal
rng_key = jax.random.PRNGKey(0)
initial_state_fn, aux_features = (di.isothermal_rest_atmosphere(
    coords=coords, physics_specs=physics_specs, p0=p0, p1=p1))
initial_state = initial_state_fn(rng_key)
ref_temps = aux_features["ref_temperatures"]
orography = di.truncated_modal_orography(aux_features["orography"], coords)
initial_state_dict, _ = di.as_dict(initial_state)
u, v = di.vor_div_to_uv_nodal(coords.horizontal, initial_state.vorticity,
                              initial_state.divergence)
initial_state_dict.update({"u": u, "v": v, "orography": orography})
nodal_steady_state_fields = di.maybe_to_nodal(initial_state_dict,
                                              coords=coords)
initial_state_ds = di.data_to_xarray(nodal_steady_state_fields,
                                     coords=coords,
                                     times=None)
temperature = di.temperature_variation_to_absolute(
    initial_state_ds.temperature_variation.data, ref_temps)
initial_state_ds = initial_state_ds.assign(
    temperature=(initial_state_ds.temperature_variation.dims, temperature))
surface_pressure = np.exp(initial_state_ds.log_surface_pressure.data[0, ...])
initial_state_ds = initial_state_ds.assign(
    surface_pressure=(initial_state_ds.log_surface_pressure.dims[1:],
                      surface_pressure))
pressure_array = initial_state_ds["surface_pressure"]
pressure_array_si = dimensionalize(pressure_array, units.pascal)
pressure_array_si.plot.imshow(x="lon", y="lat", size=5)
plt.savefig("h.00.png")
plt.close()
dt_si = 5 * units.minute
save_every = 10 * units.minute
total_time = 24 * units.hour
inner_steps = int(save_every / dt_si)
outer_steps = int(total_time / save_every)
dt = physics_specs.nondimensionalize(dt_si)
primitive = di.PrimitiveEquations(ref_temps, orography, coords, physics_specs)
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


ds = trajectory_to_xarray(coords, jax.device_get(trajectory), times)
data_array = ds["surface_pressure"] - ds["surface_pressure"].isel(time=0)
data_array_si = dimensionalize(data_array, units.pascal)
data_array_si.isel(time=slice(0, 18, 4)).plot(x="lon",
                                              y="lat",
                                              col="time",
                                              col_wrap=6)
plt.savefig("h.01.png")
plt.close()
data_array = ds["vorticity"]
data_array.isel(level=10).isel(time=slice(0, 18, 4)).plot(x="lon",
                                                          y="lat",
                                                          col="time",
                                                          col_wrap=6)
plt.savefig("h.02.png")
plt.close()
data_array = ds["total_kinetic_energy"]
data_array.plot(x="time")
ax = plt.gca()
ax.legend().remove()
plt.savefig("h.03.png")
plt.close()
hs = di.HeldSuarezForcing(coords=coords,
                          physics_specs=physics_specs,
                          reference_temperature=ref_temps,
                          p0=p0)


def ds_held_suarez_forcing(coords):
    grid = coords.horizontal
    sigma = coords.vertical.centers
    lon, _ = grid.nodal_mesh
    surface_pressure = physics_specs.nondimensionalize(p0) * np.ones_like(lon)
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


ds = ds_held_suarez_forcing(coords)


def linspace_step(start, stop, step):
    num = round((stop - start) / step) + 1
    return np.linspace(start, stop, num)


kv = ds["kv"]
kv_si = dimensionalize(kv, 1 / units.day)
kv_si.plot(size=5)
plt.savefig("h.04.png")
plt.close()
kt_array = ds["kt"]
levels = linspace_step(0, 0.3, 0.05)
kt_array_si = dimensionalize(kt_array, 1 / units.day)
p = kt_array_si.isel(lon=0).plot.contour(x="lat",
                                         y="sigma",
                                         levels=levels,
                                         size=5,
                                         aspect=1.5)
ax = plt.gca()
ax.set_ylim((1, 0))
plt.savefig("h.05.png")
plt.close()
teq_array = ds["eq_temp"]
teq_array_si = dimensionalize(teq_array, units.degK)
levels = linspace_step(205, 310, 5)
p = teq_array_si.isel(lon=0).plot.contour(x="lat",
                                          y="sigma",
                                          levels=levels,
                                          size=5,
                                          aspect=1.5)
ax = plt.gca()
ax.set_ylim((1, 0))
plt.savefig("h.06.png")
plt.close()
temperature = dimensionalize(ds["eq_temp"], units.degK)
surface_pressure = dimensionalize(ds["surface_pressure"], units.pascal)
pressure = ds.sigma * surface_pressure
kappa = di.KAPPA
potential_temperature = temperature * (pressure / p0)**-kappa
levels = linspace_step(260, 325, 5)
levels = np.concatenate(
    [levels, np.array([350, 400, 450, 500, 550, 600])], axis=0)
p = potential_temperature.isel(lon=0).plot.contour(x="lat",
                                                   y="sigma",
                                                   levels=levels,
                                                   size=5,
                                                   aspect=1.5)
ax = plt.gca()
ax.set_ylim((1, 0))
plt.colorbar(p)
print(potential_temperature.min())
print(potential_temperature.max())
plt.savefig("h.07.png")
plt.close()
dt_si = 10 * units.minute
save_every = 10 * units.day
total_time = 1200 * units.day
inner_steps = int(save_every / dt_si)
outer_steps = int(total_time / save_every)
dt = physics_specs.nondimensionalize(dt_si)
primitive = di.PrimitiveEquations(ref_temps, orography, coords, physics_specs)
hs_forcing = di.HeldSuarezForcing(coords=coords,
                                  physics_specs=physics_specs,
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
data_array = ds["total_kinetic_energy"]
data_array.plot(x="time")
ax = plt.gca()
ax.legend().remove()
plt.savefig("h.08.png")
plt.close()
mask = ds["time"] <= 150
data_array = ds["temperature"]
data_array.isel(level=-1, time=mask).plot(x="lon",
                                          y="lat",
                                          col="time",
                                          col_wrap=4)
plt.savefig("h.09.png")
plt.close()
mask = ds["time"] <= 150
data_array = ds["vorticity"]
data_array.isel(level=-1, time=mask).plot(x="lon",
                                          y="lat",
                                          col="time",
                                          col_wrap=4)
plt.savefig("h.10.png")
plt.close()
mask = ds["time"] <= 100
data_array = ds["temperature"]
data_array = dimensionalize(data_array, units.degK)
levels = linspace_step(190, 305, 5)
data_array.isel(time=mask).mean("lon").plot.contour(x="lat",
                                                    y="level",
                                                    col="time",
                                                    col_wrap=4,
                                                    levels=levels)
ax = plt.gca()
ax.set_ylim((1, 0))
plt.savefig("h.11.png")
plt.close()
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
temperature = dimensionalize(ds["temperature"], units.degK)
surface_pressure = dimensionalize(ds["surface_pressure"], units.pascal)
pressure = ds.level * surface_pressure
kappa = di.KAPPA
potential_temperature = temperature * (pressure / p0)**-kappa
levels = linspace_step(260, 325, 5)
levels = np.concatenate(
    [levels, np.array([350, 400, 450, 500, 550, 600])], axis=0)
p = (potential_temperature.isel(time=mask).mean(["lon", "time"
                                                 ]).plot.contour(x="lat",
                                                                 y="level",
                                                                 levels=levels,
                                                                 size=5,
                                                                 aspect=1.5))
ax = plt.gca()
ax.set_ylim((1, 0))
mask = ds["time"] > start_time
data_array = ds["u"]
data_array = dimensionalize(data_array, units.meter / units.s)
levels = linspace_step(-20, 28, 4)
data_array.isel(time=mask).mean(["lon", "time"]).plot.contour(x="lat",
                                                              y="level",
                                                              levels=levels,
                                                              size=5,
                                                              aspect=1.5)
ax = plt.gca()
ax.set_ylim((1, 0))
mask = ds["time"] > start_time
data_array = ds["u"]
data_array2 = ds["v"]
data_array = dimensionalize(data_array, units.meter / units.s)
data_array2 = dimensionalize(data_array2, units.meter / units.s)
mean_u = data_array.isel(time=mask).mean(["time"])
zonal_u = data_array.isel(time=mask)
mean_v = data_array2.isel(time=mask).mean(["time"])
zonal_v = data_array2.isel(time=mask)
eke = (zonal_u - mean_u)**2 + (zonal_v - mean_v)**2
p = eke.mean(["time", "lon"]).plot(x="lat", y="level", size=5, aspect=1.5)
ax = plt.gca()
ax.set_ylim((1, 0))
dt_si = 10 * units.minute
save_every = 6 * units.hours
total_time = 1 * units.week
inner_steps = int(save_every / dt_si)
outer_steps = int(total_time / save_every)
dt = physics_specs.nondimensionalize(dt_si)
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
data_array = ds["vorticity"]
data_array.thin(time=4).isel(level=-5).plot(x="lon", y="lat", col="time")
plt.savefig("h.13.png")
plt.close()
