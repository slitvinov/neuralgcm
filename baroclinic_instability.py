import functools
import jax
import numpy as np
import matplotlib.pyplot as plt
import xarray
import di

units = di.units


def steady_state_jw(
    coords,
    u0=35.0 * units.m / units.s,
    p0=1e5 * units.pascal,
    t0=288.0 * units.degK,
    delta_t=4.8e5 * units.degK,
    gamma=0.005 * units.degK / units.m,
    sigma_tropo: float = 0.2,
    sigma0: float = 0.252,
):
    u0 = di.DEFAULT_SCALE.nondimensionalize(u0)
    t0 = di.DEFAULT_SCALE.nondimensionalize(t0)
    delta_t = di.DEFAULT_SCALE.nondimensionalize(delta_t)
    p0 = di.DEFAULT_SCALE.nondimensionalize(p0)
    gamma = di.DEFAULT_SCALE.nondimensionalize(gamma)
    a = di.radius
    g = di.gravity_acceleration
    r_gas = di.ideal_gas_constant
    omega = di.angular_velocity

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
        del lon
        sigma_nu = (sigma - sigma0) * np.pi / 2
        return _get_reference_geopotential(
            sigma) + u0 * np.cos(sigma_nu)**1.5 * (
                ((-2 * np.sin(lat)**6 *
                  (np.cos(lat)**2 + 1 / 3) + 10 / 63) * u0 * np.cos(sigma_nu)**
                 (3 / 2)) +
                ((1.6 * (np.cos(lat)**3) *
                  (np.sin(lat)**2 + 2 / 3) - np.pi / 4) * a * omega))

    def _get_temperature_variation(lat, lon, sigma):
        del lon
        sigma_nu = (sigma - sigma0) * np.pi / 2
        cos_𝜎ν = np.cos(sigma_nu)
        sin_𝜎ν = np.sin(sigma_nu)
        return (0.75 * (sigma * np.pi * u0 / r_gas) * sin_𝜎ν *
                np.sqrt(cos_𝜎ν) *
                (((-2 * (np.cos(lat)**2 + 1 / 3) * np.sin(lat)**6 + 10 / 63) *
                  2 * u0 * cos_𝜎ν**(3 / 2)) +
                 ((1.6 * (np.cos(lat)**3) *
                   (np.sin(lat)**2 + 2 / 3) - np.pi / 4) * a * omega)))

    def _get_vorticity(lat, lon, sigma):
        del lon
        sigma_nu = (sigma - sigma0) * np.pi / 2
        return ((-4 * u0 / a) * (np.cos(sigma_nu)**(3 / 2)) * np.sin(lat) *
                np.cos(lat) * (2 - 5 * np.sin(lat)**2))

    def _get_surface_pressure(
        lat,
        lon,
    ):
        del lon
        return p0 * np.ones(lat.shape)[np.newaxis, ...]

    lon, sin_lat = coords.horizontal.nodal_mesh
    lat = np.arcsin(sin_lat)

    def initial_state_fn(rng_key=None):
        del rng_key
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
        state = di.State(
            vorticity=modal_vorticity,
            divergence=np.zeros_like(modal_vorticity),
            temperature_variation=coords.horizontal.to_modal(
                nodal_temperature_variation),
            log_surface_pressure=coords.horizontal.to_modal(
                log_nodal_surface_pressure),
        )
        return state

    orography = _get_geopotential(lat, lon, 1.0) / g
    geopotential = np.stack([
        _get_geopotential(lat, lon, sigma) for sigma in coords.vertical.centers
    ])
    reference_temperatures = np.stack([
        _get_reference_temperature(sigma) for sigma in coords.vertical.centers
    ])
    aux_features = {
        "geopotential": geopotential,
        "orography": orography,
        "ref_temperatures": reference_temperatures,
    }
    return initial_state_fn, aux_features


def baroclinic_perturbation_jw(
    coords,
    u_perturb=1.0 * units.m / units.s,
    lon_location=np.pi / 9,
    lat_location=2 * np.pi / 9,
    perturbation_radius=0.1,
):
    u_p = DEFAULT_SCALE.nondimensionalize(u_perturb)
    a = radius

    def _get_vorticity_perturbation(lat, lon, sigma):
        del sigma
        x = np.sin(lat_location) * np.sin(lat) + np.cos(lat_location) * np.cos(
            lat) * np.cos(lon - lon_location)
        r = a * np.arccos(x)
        R = a * perturbation_radius
        return ((u_p / a) * np.exp(-((r / R)**2)) *
                (np.tan(lat) - (2 * ((a / R)**2) * np.arccos(x)) *
                 (np.sin(lat_location) * np.cos(lat) - np.cos(lat_location) *
                  np.sin(lat) * np.cos(lon - lon_location)) /
                 (np.sqrt(1 - x**2))))

    def _get_divergence_perturbation(lat, lon, sigma):
        del sigma
        x = np.sin(lat_location) * np.sin(lat) + np.cos(lat_location) * np.cos(
            lat) * np.cos(lon - lon_location)
        r = a * np.arccos(x)
        R = a * perturbation_radius
        return ((-2 * u_p * a /
                 (R**2)) * np.exp(-((r / R)**2)) * np.arccos(x) *
                ((np.cos(lat_location) * np.sin(lon - lon_location)) /
                 (np.sqrt(1 - x**2))))

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
    state = State(
        vorticity=modal_vorticity,
        divergence=modal_divergence,
        temperature_variation=np.zeros_like(modal_vorticity),
        log_surface_pressure=np.zeros_like(modal_vorticity[:1, ...]),
    )
    return state


layers = 24
grid = di.Grid.T42()
vertical_grid = di.SigmaCoordinates.equidistant(layers)
coords = di.CoordinateSystem(grid, vertical_grid)
initial_state_fn, aux_features = steady_state_jw(coords)
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
plt.plot(f0["z_surf"])
plt.savefig("b.00.png")
plt.close()
plt.contour(f0["u"][:, 0, :])
plt.savefig("b.01.png")
plt.close()
plt.contour(f0["temperature_variation"][:, 0, :], levels=50)
plt.savefig("b.02.png")
plt.close()
plt.contour(f0["vorticity"][:, 0, :], levels=50)
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
x1 = di.data_to_xarray(f1, coords=coords, times=times)
x1["surface_pressure"] = np.exp(x1.log_surface_pressure[:, 0, :, :])
temperature = di.temperature_variation_to_absolute(
    x1.temperature_variation.data, ref_temps)
x1 = x1.assign(temperature=(x1.temperature_variation.dims, temperature))
data_array = x1["vorticity"]
data_array.isel(lon=0).thin(time=12).plot.contour(x="lat",
                                                  y="level",
                                                  col="time")
ax = plt.gca()
ax.set_ylim((1, 0))
data_array = x1["surface_pressure"] / di.DEFAULT_SCALE.nondimensionalize(
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
x1 = di.data_to_xarray(f1, coords=coords, times=times)
x1["surface_pressure"] = np.exp(x1.log_surface_pressure[:, 0, :, :])
temperature = di.temperature_variation_to_absolute(
    x1.temperature_variation.data, ref_temps)
x1 = x1.assign(temperature=(x1.temperature_variation.dims, temperature))
data_array = x1["surface_pressure"] / di.DEFAULT_SCALE.nondimensionalize(
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
data_array = x1["surface_pressure"] / di.DEFAULT_SCALE.nondimensionalize(
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
