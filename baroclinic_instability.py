import jax
import numpy as np
import matplotlib.pyplot as plt
import di

units = di.units


def temperature_variation_to_absolute(temperature_variation, ref_temperature):
    return temperature_variation + ref_temperature[:, np.newaxis, np.newaxis]


def get_reference_temperature(sigma):
    top_mean_t = t0 * sigma**(r_gas * gamma / g)
    if sigma < sigma_tropo:
        return top_mean_t + delta_t * (sigma_tropo - sigma)**5
    else:
        return top_mean_t


def get_reference_geopotential(sigma):
    top_mean_potential = (t0 * g / gamma) * (1 - sigma**(r_gas * gamma / g))
    if sigma < sigma_tropo:
        return top_mean_potential - r_gas * delta_t * (
            (np.log(sigma / sigma_tropo) + 137 / 60) * sigma_tropo**5 -
            5 * sigma * sigma_tropo**4 + 5 * (sigma**2) * (sigma_tropo**3) -
            (10 / 3) * (sigma_tropo**2) * sigma**3 +
            (5 / 4) * sigma_tropo * sigma**4 - (sigma**5) / 5)
    else:
        return top_mean_potential


def get_geopotential(lat, lon, sigma):
    del lon
    sigma_nu = (sigma - sigma0) * np.pi / 2
    return get_reference_geopotential(sigma) + u0 * np.cos(sigma_nu)**1.5 * (
        ((-2 * np.sin(lat)**6 *
          (np.cos(lat)**2 + 1 / 3) + 10 / 63) * u0 * np.cos(sigma_nu)**
         (3 / 2)) + ((1.6 * (np.cos(lat)**3) *
                      (np.sin(lat)**2 + 2 / 3) - np.pi / 4) * 0.5))


def get_temperature_variation(lat, lon, sigma):
    del lon
    sigma_nu = (sigma - sigma0) * np.pi / 2
    cos_𝜎ν = np.cos(sigma_nu)
    sin_𝜎ν = np.sin(sigma_nu)
    return (0.75 * (sigma * np.pi * u0 / r_gas) * sin_𝜎ν * np.sqrt(cos_𝜎ν) *
            (((-2 * (np.cos(lat)**2 + 1 / 3) * np.sin(lat)**6 + 10 / 63) * 2 *
              u0 * cos_𝜎ν**(3 / 2)) +
             ((1.6 * (np.cos(lat)**3) *
               (np.sin(lat)**2 + 2 / 3) - np.pi / 4) * 0.5)))


def get_vorticity(lat, lon, sigma):
    del lon
    sigma_nu = (sigma - sigma0) * np.pi / 2
    return ((-4 * u0) * (np.cos(sigma_nu)**(3 / 2)) * np.sin(lat) *
            np.cos(lat) * (2 - 5 * np.sin(lat)**2))


def get_surface_pressure(lat, lon):
    del lon
    return p0 * np.ones(lat.shape)[np.newaxis, ...]


def get_vorticity_perturbation(lat, lon, sigma):
    del sigma
    x = np.sin(lat_location) * np.sin(lat) + np.cos(lat_location) * np.cos(
        lat) * np.cos(lon - lon_location)
    r = np.arccos(x)
    R = perturbation_radius
    return (
        u_p * np.exp(-((r / R)**2)) *
        (np.tan(lat) - (2 * ((1.0 / R)**2) * np.arccos(x)) *
         (np.sin(lat_location) * np.cos(lat) -
          np.cos(lat_location) * np.sin(lat) * np.cos(lon - lon_location)) /
         (np.sqrt(1 - x**2))))


def get_divergence_perturbation(lat, lon, sigma):
    del sigma
    x = np.sin(lat_location) * np.sin(lat) + np.cos(lat_location) * np.cos(
        lat) * np.cos(lon - lon_location)
    r = np.arccos(x)
    R = perturbation_radius
    return ((-2 * u_p / (R**2)) * np.exp(-((r / R)**2)) * np.arccos(x) *
            ((np.cos(lat_location) * np.sin(lon - lon_location)) /
             (np.sqrt(1 - x**2))))


layers = 12
grid = di.Grid(longitude_wavenumbers=22,
               total_wavenumbers=23,
               longitude_nodes=64,
               latitude_nodes=32)
vertical_grid = di.SigmaCoordinates(
    np.linspace(0, 1, layers + 1, dtype=np.float32))

u0 = 35.0 * units.m / units.s
p0 = 1e5 * units.pascal
t0 = 288.0 * units.degK
delta_t = 4.8e5 * units.degK
gamma = 0.005 * units.degK / units.m
sigma_tropo = 0.2
sigma0 = 0.252
coords = di.CoordinateSystem(grid, vertical_grid)
u0 = di.DEFAULT_SCALE.nondimensionalize(u0)
t0 = di.DEFAULT_SCALE.nondimensionalize(t0)
delta_t = di.DEFAULT_SCALE.nondimensionalize(delta_t)
p0 = di.DEFAULT_SCALE.nondimensionalize(p0)
gamma = di.DEFAULT_SCALE.nondimensionalize(gamma)
g = di.gravity_acceleration
r_gas = di.ideal_gas_constant
lon, sin_lat = coords.horizontal.nodal_mesh
lat = np.arcsin(sin_lat)
orography = get_geopotential(lat, lon, 1.0) / g
geopotential = np.stack(
    [get_geopotential(lat, lon, sigma) for sigma in coords.vertical.centers])
reference_temperatures = np.stack(
    [get_reference_temperature(sigma) for sigma in coords.vertical.centers])
aux_features = {
    "geopotential": geopotential,
    "orography": orography,
    "ref_temperatures": reference_temperatures,
}
nodal_vorticity = np.stack(
    [get_vorticity(lat, lon, sigma) for sigma in coords.vertical.centers])
modal_vorticity = coords.horizontal.to_modal(nodal_vorticity)
nodal_temperature_variation = np.stack([
    get_temperature_variation(lat, lon, sigma)
    for sigma in coords.vertical.centers
])
log_nodal_surface_pressure = np.log(get_surface_pressure(lat, lon))
steady_state = di.State(
    vorticity=modal_vorticity,
    divergence=np.zeros_like(modal_vorticity),
    temperature_variation=coords.horizontal.to_modal(
        nodal_temperature_variation),
    log_surface_pressure=coords.horizontal.to_modal(
        log_nodal_surface_pressure),
)

ref_temps = aux_features["ref_temperatures"]
orography = di.truncated_modal_orography(aux_features["orography"], coords)
steady_state_dict = steady_state.asdict()
u, v = di.vor_div_to_uv_nodal(grid, steady_state.vorticity,
                              steady_state.divergence)
steady_state_dict.update({"u": u, "v": v, "z_surf": orography})
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
trajectory_dict = trajectory.asdict()
u, v = di.vor_div_to_uv_nodal(grid, trajectory.vorticity,
                              trajectory.divergence)
trajectory_dict.update({"u": u, "v": v})
u_perturb = 1.0 * units.m / units.s
lon_location = np.pi / 9
lat_location = 2 * np.pi / 9
perturbation_radius = 0.1
u_p = di.DEFAULT_SCALE.nondimensionalize(u_perturb)
lon, sin_lat = coords.horizontal.nodal_mesh
lat = np.arcsin(sin_lat)
nodal_vorticity = np.stack([
    get_vorticity_perturbation(lat, lon, sigma)
    for sigma in coords.vertical.centers
])
nodal_divergence = np.stack([
    get_divergence_perturbation(lat, lon, sigma)
    for sigma in coords.vertical.centers
])
modal_vorticity = coords.horizontal.to_modal(nodal_vorticity)
modal_divergence = coords.horizontal.to_modal(nodal_divergence)
perturbation = di.State(
    vorticity=modal_vorticity,
    divergence=modal_divergence,
    temperature_variation=np.zeros_like(modal_vorticity),
    log_surface_pressure=np.zeros_like(modal_vorticity[:1, ...]),
)

state = steady_state + perturbation
save_every = 2 * units.hour
total_time = 2 * units.week
inner_steps = int(save_every / dt_s)
outer_steps = int(total_time / save_every)
integrate_fn = di.trajectory_from_step(step_fn, outer_steps, inner_steps)
integrate_fn = jax.jit(integrate_fn)
final, trajectory = jax.block_until_ready(integrate_fn(state))
trajectory = jax.device_get(trajectory)
trajectory_dict = trajectory.asdict()
u, v = di.vor_div_to_uv_nodal(grid, trajectory.vorticity,
                              trajectory.divergence)
trajectory_dict.update({"u": u, "v": v})
f1 = di.maybe_to_nodal(trajectory_dict, coords=coords)
temperature = temperature_variation_to_absolute(f1["temperature_variation"],
                                                ref_temps)
levels = [(220 + 10 * i) for i in range(10)]
plt.contourf(temperature[119, 22, :, :], levels=levels, cmap=plt.cm.Spectral_r)
plt.savefig("b.09.png")
np.asarray(temperature).tofile("b.09.raw")
plt.close()
