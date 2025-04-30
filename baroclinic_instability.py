import functools
import jax
import numpy as np
import matplotlib.pyplot as plt
import xarray
import di

units = di.units
layers = 24
grid = di.Grid.T42()
vertical_grid = di.SigmaCoordinates.equidistant(layers)
coords = di.CoordinateSystem(grid, vertical_grid)
initial_state_fn, aux_features = di.steady_state_jw(coords)
steady_state = initial_state_fn()
ref_temps = aux_features["ref_temperatures"]
orography = di.truncated_modal_orography(aux_features["orography"], coords)


def dimensionalize(x, unit):
    dimensionalize = functools.partial(di.DEFAULT_SCALE.dimensionalize,
                                       unit=unit)
    return xarray.apply_ufunc(dimensionalize, x)


steady_state_dict, _ = di.as_dict(steady_state)
u, v = di.vor_div_to_uv_nodal(grid, steady_state.vorticity,
                              steady_state.divergence)
steady_state_dict.update({"u": u, "v": v, "z_surf": orography})
f0 = di.maybe_to_nodal(steady_state_dict, coords=coords)
x0 = di.data_to_xarray(f0,
                                     coords=coords,
                                     times=None)
temperature = di.temperature_variation_to_absolute(
    x0.temperature_variation.data, ref_temps)
x0 = x0.assign(
    temperature=(x0.temperature_variation.dims, temperature))
phi = x0["z_surf"] * di.gravity_acceleration
phi_si = dimensionalize(phi, units.m**2 / units.s**2)
phi_si.isel(lon=0).plot(x="lat")
plt.savefig("b.00.png")
plt.close()
u_array = x0["u"]
u_array_si = dimensionalize(u_array, units.m / units.s)
levels = [3 * i for i in range(1, 12)]
u_array_si.isel(lon=0).plot.contour(x="lat", y="level", levels=levels)
ax = plt.gca()
ax.set_ylim((1, 0))
plt.savefig("b.01.png")
plt.close()
t_array = x0["temperature"]
t_array_si = dimensionalize(t_array, units.degK)
levels = np.linspace(210, 305, 1 + (305 - 210) // 5)
t_array_si.isel(lon=0).plot.contour(x="lat", y="level", levels=levels)
ax = plt.gca()
ax.set_ylim((1, 0))
plt.savefig("b.02.png")
plt.close()
voriticty_array = x0["vorticity"]
voriticty_array_si = dimensionalize(voriticty_array, 1 / units.s)
levels = np.linspace(-1.75e-5, 1.75e-5, 15)
voriticty_array_si.isel(lon=0).plot.contour(x="lat", y="level", levels=levels)
ax = plt.gca()
ax.set_ylim((1, 0))
plt.savefig("b.03.png")
plt.close()
primitive = di.PrimitiveEquations(ref_temps, orography, coords)
dt_s = 100 * units.s
dt = di.DEFAULT_SCALE.nondimensionalize(dt_s)
step_fn = di.imex_rk_sil3(primitive, dt)
save_every = 2 * units.hour
total_time = 1 * units.week
inner_steps = int(save_every / dt_s)
outer_steps = int(total_time / save_every)
filters = [
    di.exponential_step_filter(grid, dt),
]
step_fn = di.step_with_filters(step_fn, filters)
integrate_fn = di.trajectory_from_step(step_fn, outer_steps, inner_steps)
integrate_fn = jax.jit(integrate_fn)
final, trajectory = jax.block_until_ready(integrate_fn(steady_state))
trajectory = jax.device_get(trajectory)
times = save_every * np.arange(outer_steps)
trajectory_dict, _ = di.as_dict(trajectory)
u, v = di.vor_div_to_uv_nodal(grid, trajectory.vorticity,
                              trajectory.divergence)
trajectory_dict.update({"u": u, "v": v})
f1 = di.maybe_to_nodal(trajectory_dict, coords=coords)
x1 = di.data_to_xarray(f1,
                                  coords=coords,
                                  times=times)
x1["surface_pressure"] = np.exp(
    x1.log_surface_pressure[:, 0, :, :])
temperature = di.temperature_variation_to_absolute(
    x1.temperature_variation.data, ref_temps)
x1 = x1.assign(
    temperature=(x1.temperature_variation.dims, temperature))
data_array = x1["vorticity"]
data_array.isel(lon=0).thin(time=12).plot.contour(x="lat",
                                                  y="level",
                                                  col="time")
ax = plt.gca()
ax.set_ylim((1, 0))
data_array = x1[
    "surface_pressure"] / di.DEFAULT_SCALE.nondimensionalize(
        1e5 * units.pascal)
data_array.max(["lon"]).plot(x="time", hue="lat")
ax = plt.gca()
ax.legend().remove()
plt.savefig("b.04.png")
plt.close()
t_array = x1["temperature"]
t_array_si = dimensionalize(t_array, units.degK)
levels = np.linspace(210, 305, 1 + (305 - 210) // 5)
t_array_si.isel(lon=0).thin(time=12).plot.contour(x="lat",
                                                  y="level",
                                                  levels=levels,
                                                  col="time")
ax = plt.gca()
ax.set_ylim((1, 0))
plt.savefig("b.05.png")
plt.close()
data_array = x1["divergence"]
data_array.mean(["lat", "lon"]).plot(x="time", hue="level")
ax = plt.gca()
plt.savefig("b.06.png")
plt.close()
perturbation = di.baroclinic_perturbation_jw(coords)
state = steady_state + perturbation
save_every = 2 * units.hour
total_time = 2 * units.week
inner_steps = int(save_every / dt_s)
outer_steps = int(total_time / save_every)
integrate_fn = di.trajectory_from_step(step_fn, outer_steps, inner_steps)
integrate_fn = jax.jit(integrate_fn)
final, trajectory = jax.block_until_ready(integrate_fn(state))
trajectory = jax.device_get(trajectory)
times = (save_every * np.arange(outer_steps)).to(units.s)
trajectory_dict, _ = di.as_dict(trajectory)
u, v = di.vor_div_to_uv_nodal(grid, trajectory.vorticity,
                              trajectory.divergence)
trajectory_dict.update({"u": u, "v": v})
f1 = di.maybe_to_nodal(trajectory_dict, coords=coords)
x1 = di.data_to_xarray(f1,
                                  coords=coords,
                                  times=times)
x1["surface_pressure"] = np.exp(
    x1.log_surface_pressure[:, 0, :, :])
temperature = di.temperature_variation_to_absolute(
    x1.temperature_variation.data, ref_temps)
x1 = x1.assign(
    temperature=(x1.temperature_variation.dims, temperature))
data_array = x1[
    "surface_pressure"] / di.DEFAULT_SCALE.nondimensionalize(
        1e5 * units.pascal)
levels = [(992 + 2 * i) / 1000 for i in range(8)]
data_array.sel({
    "lat":
    slice(0, 90),
    "lon":
    slice(45, 360),
    "time": [(4 * units.day).to(units.s).m, (6 * units.day).to(units.s).m],
}).plot.contourf(x="lon",
                 y="lat",
                 row="time",
                 levels=levels,
                 cmap=plt.cm.Spectral_r)
fig = plt.gcf()
fig.set_figwidth(10)
plt.savefig("b.07.png")
plt.close()
data_array = x1[
    "surface_pressure"] / di.DEFAULT_SCALE.nondimensionalize(
        1e5 * units.pascal)
levels = [(930 + 10 * i) / 1000 for i in range(10)]
(data_array.sel({
    "lat":
    slice(0, 90),
    "lon":
    slice(45, 360),
    "time": [(8 * units.day).to(units.s).m, (10 * units.day).to(units.s).m],
}).plot.contourf(x="lon",
                 y="lat",
                 row="time",
                 levels=levels,
                 cmap=plt.cm.Spectral_r))
fig = plt.gcf()
fig.set_figwidth(10)
plt.savefig("b.08.png")
plt.close()
temp_array = x1["temperature"]
levels = [(220 + 10 * i) for i in range(10)]
target_pressure = 0.85 * di.DEFAULT_SCALE.nondimensionalize(1e5 * units.pascal)
(temp_array.sel({
    "lat":
    slice(0, 90),
    "lon":
    slice(45, 360),
    "time": [
        (4 * units.day).to(units.s).m,
        (6 * units.day).to(units.s).m,
        (8 * units.day).to(units.s).m,
        (10 * units.day).to(units.s).m,
    ],
}).isel(level=22).plot.contourf(x="lon",
                                y="lat",
                                row="time",
                                levels=levels,
                                cmap=plt.cm.Spectral_r))
fig = plt.gcf()
fig.set_figwidth(12)
plt.savefig("b.09.png")
plt.close()
voriticty_array = x1["vorticity"]
target_pressure = 0.85 * di.DEFAULT_SCALE.nondimensionalize(1e5 * units.pascal)
voriticty_array_si = dimensionalize(voriticty_array, 1 / units.s)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
levels = [-3e-5 + 1e-5 * i for i in range(10)]
(voriticty_array_si.sel({
    "lat": slice(25, 75),
    "lon": slice(90, 210),
    "time": (7 * units.day).to(units.s).m,
}).isel(level=22).plot.contourf(x="lon", y="lat", levels=levels, ax=ax1))
levels = [-10e-5 + 5e-5 * i for i in range(11)]
(voriticty_array_si.sel({
    "lat": slice(25, 75),
    "lon": slice(120, 270),
    "time": (9 * units.day).to(units.s).m,
}).isel(level=22).plot.contourf(x="lon", y="lat", levels=levels, ax=ax2))
fig.set_figwidth(25)
plt.savefig("b.10.png")
plt.close()
times = ((np.arange(12) * units.day).to(units.second)).astype(np.int32)
data = temp_array.sel(lat=slice(54, 56), lon=slice(120, 270),
                      time=times).isel(lat=0)
data.attrs["units"] = "seconds"
data.plot.contourf(x="lon", y="level", row="time", aspect=2, col_wrap=3)
plt.savefig("b.11.png")
plt.close()
