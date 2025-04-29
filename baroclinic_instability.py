import functools
import dinosaur
import jax
import numpy as np
import matplotlib.pyplot as plt
import xarray
import dinosaur

units = dinosaur.scales.units
layers = 24
grid = dinosaur.spherical_harmonic.Grid.T42()
vertical_grid = dinosaur.sigma_coordinates.SigmaCoordinates.equidistant(layers)
coords = dinosaur.coordinate_systems.CoordinateSystem(grid, vertical_grid)
physics_specs = dinosaur.primitive_equations.PrimitiveEquationsSpecs.from_si()
initial_state_fn, aux_features = dinosaur.primitive_equations_states.steady_state_jw(
    coords, physics_specs)
steady_state = initial_state_fn()
ref_temps = aux_features['ref_temperatures']
orography = dinosaur.primitive_equations.truncated_modal_orography(
    aux_features['orography'], coords)


def dimensionalize(x, unit):
    """Dimensionalizes `xarray.DataArray`s."""
    dimensionalize = functools.partial(physics_specs.dimensionalize, unit=unit)
    return xarray.apply_ufunc(dimensionalize, x)


steady_state_dict, _ = dinosaur.pytree_utils.as_dict(steady_state)
u, v = dinosaur.spherical_harmonic.vor_div_to_uv_nodal(grid,
                                                       steady_state.vorticity,
                                                       steady_state.divergence)
steady_state_dict.update({'u': u, 'v': v, 'z_surf': orography})
nodal_steady_state_fields = dinosaur.coordinate_systems.maybe_to_nodal(
    steady_state_dict, coords=coords)
initial_state_ds = dinosaur.xarray_utils.data_to_xarray(
    nodal_steady_state_fields, coords=coords, times=None)
temperature = dinosaur.xarray_utils.temperature_variation_to_absolute(
    initial_state_ds.temperature_variation.data, ref_temps)
initial_state_ds = initial_state_ds.assign(
    temperature=(initial_state_ds.temperature_variation.dims, temperature))
phi = initial_state_ds['z_surf'] * physics_specs.g
phi_si = dimensionalize(phi, units.m**2 / units.s**2)
phi_si.isel(lon=0).plot(x='lat')
plt.savefig("00.png")
plt.close()
u_array = initial_state_ds['u']
u_array_si = dimensionalize(u_array, units.m / units.s)
levels = [3 * i for i in range(1, 12)]
u_array_si.isel(lon=0).plot.contour(x='lat', y='level', levels=levels)
ax = plt.gca()
ax.set_ylim((1, 0))
plt.savefig("01.png")
plt.close()
t_array = initial_state_ds['temperature']
t_array_si = dimensionalize(t_array, units.degK)
levels = np.linspace(210, 305, 1 + (305 - 210) // 5)
t_array_si.isel(lon=0).plot.contour(x='lat', y='level', levels=levels)
ax = plt.gca()
ax.set_ylim((1, 0))
plt.savefig("02.png")
plt.close()
voriticty_array = initial_state_ds['vorticity']
voriticty_array_si = dimensionalize(voriticty_array, 1 / units.s)
levels = np.linspace(-1.75e-5, 1.75e-5, 15)
voriticty_array_si.isel(lon=0).plot.contour(x='lat', y='level', levels=levels)
ax = plt.gca()
ax.set_ylim((1, 0))
plt.savefig("03.png")
plt.close()
primitive = dinosaur.primitive_equations.PrimitiveEquations(
    ref_temps, orography, coords, physics_specs)
dt_s = 100 * units.s
dt = physics_specs.nondimensionalize(dt_s)
step_fn = dinosaur.time_integration.imex_rk_sil3(primitive, dt)
save_every = 2 * units.hour
total_time = 1 * units.week
inner_steps = int(save_every / dt_s)
outer_steps = int(total_time / save_every)
filters = [
    dinosaur.time_integration.exponential_step_filter(grid, dt),
]
step_fn = dinosaur.time_integration.step_with_filters(step_fn, filters)
integrate_fn = dinosaur.time_integration.trajectory_from_step(
    step_fn, outer_steps, inner_steps)
integrate_fn = jax.jit(integrate_fn)
final, trajectory = jax.block_until_ready(integrate_fn(steady_state))
trajectory = jax.device_get(trajectory)
times = save_every * np.arange(outer_steps)
trajectory_dict, _ = dinosaur.pytree_utils.as_dict(trajectory)
u, v = dinosaur.spherical_harmonic.vor_div_to_uv_nodal(grid,
                                                       trajectory.vorticity,
                                                       trajectory.divergence)
trajectory_dict.update({'u': u, 'v': v})
nodal_trajectory_fields = dinosaur.coordinate_systems.maybe_to_nodal(
    trajectory_dict, coords=coords)
trajectory_ds = dinosaur.xarray_utils.data_to_xarray(nodal_trajectory_fields,
                                                     coords=coords,
                                                     times=times)
trajectory_ds['surface_pressure'] = np.exp(
    trajectory_ds.log_surface_pressure[:, 0, :, :])
temperature = dinosaur.xarray_utils.temperature_variation_to_absolute(
    trajectory_ds.temperature_variation.data, ref_temps)
trajectory_ds = trajectory_ds.assign(
    temperature=(trajectory_ds.temperature_variation.dims, temperature))
data_array = trajectory_ds['vorticity']
data_array.isel(lon=0).thin(time=12).plot.contour(x='lat',
                                                  y='level',
                                                  col='time')
ax = plt.gca()
ax.set_ylim((1, 0))
data_array = (trajectory_ds['surface_pressure'] /
              physics_specs.nondimensionalize(1e5 * units.pascal))
data_array.max(['lon']).plot(x='time', hue='lat')
ax = plt.gca()
ax.legend().remove()
plt.savefig("04.png")
plt.close()
t_array = trajectory_ds['temperature']
t_array_si = dimensionalize(t_array, units.degK)
levels = np.linspace(210, 305, 1 + (305 - 210) // 5)
t_array_si.isel(lon=0).thin(time=12).plot.contour(x='lat',
                                                  y='level',
                                                  levels=levels,
                                                  col='time')
ax = plt.gca()
ax.set_ylim((1, 0))
plt.savefig("05.png")
plt.close()
data_array = trajectory_ds['divergence']
data_array.mean(['lat', 'lon']).plot(x='time', hue='level')
ax = plt.gca()
plt.savefig("06.png")
plt.close()
perturbation = dinosaur.primitive_equations_states.baroclinic_perturbation_jw(
    coords, physics_specs)
state = steady_state + perturbation
save_every = 2 * units.hour
total_time = 2 * units.week
inner_steps = int(save_every / dt_s)
outer_steps = int(total_time / save_every)
integrate_fn = dinosaur.time_integration.trajectory_from_step(
    step_fn, outer_steps, inner_steps)
integrate_fn = jax.jit(integrate_fn)
final, trajectory = jax.block_until_ready(integrate_fn(state))
trajectory = jax.device_get(trajectory)
times = (save_every * np.arange(outer_steps)).to(units.s)
trajectory_dict, _ = dinosaur.pytree_utils.as_dict(trajectory)
u, v = dinosaur.spherical_harmonic.vor_div_to_uv_nodal(grid,
                                                       trajectory.vorticity,
                                                       trajectory.divergence)
trajectory_dict.update({'u': u, 'v': v})
nodal_trajectory_fields = dinosaur.coordinate_systems.maybe_to_nodal(
    trajectory_dict, coords=coords)
trajectory_ds = dinosaur.xarray_utils.data_to_xarray(nodal_trajectory_fields,
                                                     coords=coords,
                                                     times=times)
trajectory_ds['surface_pressure'] = np.exp(
    trajectory_ds.log_surface_pressure[:, 0, :, :])
temperature = dinosaur.xarray_utils.temperature_variation_to_absolute(
    trajectory_ds.temperature_variation.data, ref_temps)
trajectory_ds = trajectory_ds.assign(
    temperature=(trajectory_ds.temperature_variation.dims, temperature))
data_array = (trajectory_ds['surface_pressure'] /
              physics_specs.nondimensionalize(1e5 * units.pascal))
levels = [(992 + 2 * i) / 1000 for i in range(8)]
data_array.sel({
    'lat':
    slice(0, 90),
    'lon':
    slice(45, 360),
    'time': [(4 * units.day).to(units.s).m, (6 * units.day).to(units.s).m],
}).plot.contourf(x='lon',
                 y='lat',
                 row='time',
                 levels=levels,
                 cmap=plt.cm.Spectral_r)
fig = plt.gcf()
fig.set_figwidth(10)
plt.savefig("07.png")
plt.close()
data_array = (trajectory_ds['surface_pressure'] /
              physics_specs.nondimensionalize(1e5 * units.pascal))
levels = [(930 + 10 * i) / 1000 for i in range(10)]
(data_array.sel({
    'lat':
    slice(0, 90),
    'lon':
    slice(45, 360),
    'time': [(8 * units.day).to(units.s).m, (10 * units.day).to(units.s).m]
}).plot.contourf(x='lon',
                 y='lat',
                 row='time',
                 levels=levels,
                 cmap=plt.cm.Spectral_r))
fig = plt.gcf()
fig.set_figwidth(10)
plt.savefig("08.png")
plt.close()
temp_array = trajectory_ds['temperature']
levels = [(220 + 10 * i) for i in range(10)]
target_pressure = 0.85 * physics_specs.nondimensionalize(1e5 * units.pascal)
(temp_array.sel({
    'lat':
    slice(0, 90),
    'lon':
    slice(45, 360),
    'time': [(4 * units.day).to(units.s).m, (6 * units.day).to(units.s).m,
             (8 * units.day).to(units.s).m, (10 * units.day).to(units.s).m]
}).isel(level=22).plot.contourf(x='lon',
                                y='lat',
                                row='time',
                                levels=levels,
                                cmap=plt.cm.Spectral_r))
fig = plt.gcf()
fig.set_figwidth(12)
plt.savefig("09.png")
plt.close()
voriticty_array = trajectory_ds['vorticity']
target_pressure = 0.85 * physics_specs.nondimensionalize(1e5 * units.pascal)
voriticty_array_si = dimensionalize(voriticty_array, 1 / units.s)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
levels = [-3e-5 + 1e-5 * i for i in range(10)]
(voriticty_array_si.sel({
    'lat': slice(25, 75),
    'lon': slice(90, 210),
    'time': (7 * units.day).to(units.s).m
}).isel(level=22).plot.contourf(x='lon', y='lat', levels=levels, ax=ax1))
levels = [-10e-5 + 5e-5 * i for i in range(11)]
(voriticty_array_si.sel({
    'lat': slice(25, 75),
    'lon': slice(120, 270),
    'time': (9 * units.day).to(units.s).m
}).isel(level=22).plot.contourf(x='lon', y='lat', levels=levels, ax=ax2))
fig.set_figwidth(25)
plt.savefig("10.png")
plt.close()
times = ((np.arange(12) * units.day).to(units.second)).astype(np.int32)
data = (temp_array.sel(lat=slice(54, 56), lon=slice(120, 270),
                       time=times).isel(lat=0))
data.attrs['units'] = 'seconds'
data.plot.contourf(x='lon', y='level', row='time', aspect=2, col_wrap=3)
plt.savefig("11.png")
plt.close()
