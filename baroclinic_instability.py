import jax
import numpy as np
import matplotlib.pyplot as plt
import di
import scipy.special
import jax.numpy as jnp


def get_reference_temperature(sigma):
    top_mean_t = t0 * sigma**(r_gas * gamma / gravity_acceleration)
    if sigma < sigma_tropo:
        return top_mean_t + delta_t * (sigma_tropo - sigma)**5
    else:
        return top_mean_t


def get_reference_geopotential(sigma):
    top_mean_potential = (t0 * gravity_acceleration / gamma) * (
        1 - sigma**(r_gas * gamma / gravity_acceleration))
    if sigma < sigma_tropo:
        return top_mean_potential - r_gas * delta_t * (
            (np.log(sigma / sigma_tropo) + 137 / 60) * sigma_tropo**5 -
            5 * sigma * sigma_tropo**4 + 5 * (sigma**2) * (sigma_tropo**3) -
            (10 / 3) * (sigma_tropo**2) * sigma**3 +
            (5 / 4) * sigma_tropo * sigma**4 - (sigma**5) / 5)
    else:
        return top_mean_potential


def get_geopotential(lat, sigma):
    sigma_nu = (sigma - sigma0) * np.pi / 2
    return get_reference_geopotential(sigma) + u0 * np.cos(sigma_nu)**1.5 * (
        ((-2 * np.sin(lat)**6 *
          (np.cos(lat)**2 + 1 / 3) + 10 / 63) * u0 * np.cos(sigma_nu)**
         (3 / 2)) + ((1.6 * (np.cos(lat)**3) *
                      (np.sin(lat)**2 + 2 / 3) - np.pi / 4) * 0.5))


def get_temperature_variation(lat, sigma):
    sigma_nu = (sigma - sigma0) * np.pi / 2
    cos_𝜎ν = np.cos(sigma_nu)
    sin_𝜎ν = np.sin(sigma_nu)
    return (0.75 * (sigma * np.pi * u0 / r_gas) * sin_𝜎ν * np.sqrt(cos_𝜎ν) *
            (((-2 * (np.cos(lat)**2 + 1 / 3) * np.sin(lat)**6 + 10 / 63) * 2 *
              u0 * cos_𝜎ν**(3 / 2)) +
             ((1.6 * (np.cos(lat)**3) *
               (np.sin(lat)**2 + 2 / 3) - np.pi / 4) * 0.5)))


def get_vorticity(lat, sigma):
    sigma_nu = (sigma - sigma0) * np.pi / 2
    return ((-4 * u0) * (np.cos(sigma_nu)**(3 / 2)) * np.sin(lat) *
            np.cos(lat) * (2 - 5 * np.sin(lat)**2))


def get_vorticity_perturbation(lat, lon):
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


def get_divergence_perturbation(lat, lon):
    x = np.sin(lat_location) * np.sin(lat) + np.cos(lat_location) * np.cos(
        lat) * np.cos(lon - lon_location)
    r = np.arccos(x)
    R = perturbation_radius
    return ((-2 * u_p / (R**2)) * np.exp(-((r / R)**2)) * np.arccos(x) *
            ((np.cos(lat_location) * np.sin(lon - lon_location)) /
             (np.sqrt(1 - x**2))))


def to_modal(z):
    return di.tree_map_over_nonscalars(transform, z)


def to_nodal(x):
    return di.tree_map_over_nonscalars(inverse_transform, x)


def transform(x):
    f, p, w = basis()
    wx = w * x
    fwx = di.einsum("im,...ij->...mj", f, wx)
    pfwx = di.einsum("mjl,...mj->...ml", p, fwx)
    return pfwx


def inverse_transform(x):
    f, p, w = basis()
    px = di.einsum("mjl,...ml->...mj", p, x)
    fpx = di.einsum("im,...mj->...ij", f, px)
    return fpx


def basis():
    f = di.real_basis(
        wavenumbers=longitude_wavenumbers,
        nodes=longitude_nodes,
    )
    wf = 2 * np.pi / longitude_nodes
    x, wp = scipy.special.roots_legendre(latitude_nodes)
    w = wf * wp
    p = di.evaluate(n_m=longitude_wavenumbers, n_l=total_wavenumbers, x=x)
    p = np.repeat(p, 2, axis=0)
    p = p[1:]
    return f, p, w


def clip_wavenumbers(x):

    def clip(x):
        modal_shape = 2 * longitude_wavenumbers - 1, total_wavenumbers
        mask = jnp.ones(modal_shape[-1], x.dtype).at[-1:].set(0)
        return x * mask

    return di.tree_map_over_nonscalars(clip, x)


layers = 12
gravity_acceleration = 7.2364082834567185e+01
sigma_tropo = 0.2
sigma0 = 0.252
u0 = 0.03766767260790817
t0 = 288.0
delta_t = 4.8e5
p0 = 2.995499768455064e+19
gamma = 31856.1
kappa = 2 / 7
r_gas = kappa * 0.0011628807950492582
dt = 0.014584
perturbation_radius = 0.1
u_p = 1.0762192173688048e-03
longitude_wavenumbers = 22
total_wavenumbers = 23
longitude_nodes = 64
latitude_nodes = 32
grid = di.Grid(longitude_wavenumbers=longitude_wavenumbers,
               total_wavenumbers=total_wavenumbers,
               longitude_nodes=longitude_nodes,
               latitude_nodes=latitude_nodes)
vertical_grid = di.SigmaCoordinates(
    np.linspace(0, 1, layers + 1, dtype=np.float32))
coords = di.CoordinateSystem(grid, vertical_grid)
longitude = np.linspace(0, 2 * np.pi, longitude_nodes, endpoint=False)
sin_latitude, _ = scipy.special.roots_legendre(latitude_nodes)
lon, sin_lat = np.meshgrid(longitude, sin_latitude, indexing="ij")
lat = np.arcsin(sin_lat)
geopotential = np.stack(
    [get_geopotential(lat, sigma) for sigma in vertical_grid.centers])
reference_temperatures = np.stack(
    [get_reference_temperature(sigma) for sigma in vertical_grid.centers])
nodal_vorticity = np.stack(
    [get_vorticity(lat, sigma) for sigma in vertical_grid.centers])
modal_vorticity = to_modal(nodal_vorticity)
nodal_temperature_variation = np.stack(
    [get_temperature_variation(lat, sigma) for sigma in vertical_grid.centers])
log_nodal_surface_pressure = np.log(p0 * np.ones(lat.shape)[np.newaxis, ...])
steady_state = di.State(
    vorticity=modal_vorticity,
    divergence=np.zeros_like(modal_vorticity),
    temperature_variation=to_modal(nodal_temperature_variation),
    log_surface_pressure=to_modal(log_nodal_surface_pressure),
)
orography = get_geopotential(lat, 1.0) / gravity_acceleration
orography = clip_wavenumbers(to_modal(orography))
primitive = di.PrimitiveEquations(reference_temperatures, orography, coords)
step_fn = di.imex_runge_kutta(primitive, dt)
filters = [
    di.exponential_step_filter(total_wavenumbers, dt),
]
step_fn = di.step_with_filters(step_fn, filters)
inner_steps = 72
outer_steps = 84
integrate_fn = di.trajectory_from_step(step_fn, outer_steps, inner_steps)
integrate_fn = jax.jit(integrate_fn)
final, trajectory = jax.block_until_ready(integrate_fn(steady_state))
trajectory = jax.device_get(trajectory)
lon_location = np.pi / 9
lat_location = 2 * np.pi / 9
nodal_vorticity = np.stack(
    [get_vorticity_perturbation(lat, lon) for sigma in vertical_grid.centers])
nodal_divergence = np.stack(
    [get_divergence_perturbation(lat, lon) for sigma in vertical_grid.centers])
modal_vorticity = to_modal(nodal_vorticity)
modal_divergence = to_modal(nodal_divergence)
perturbation = di.State(
    vorticity=modal_vorticity,
    divergence=modal_divergence,
    temperature_variation=np.zeros_like(modal_vorticity),
    log_surface_pressure=np.zeros_like(modal_vorticity[:1, ...]),
)

state = steady_state + perturbation
outer_steps = 168
integrate_fn = di.trajectory_from_step(step_fn, outer_steps, inner_steps)
integrate_fn = jax.jit(integrate_fn)
final, trajectory = jax.block_until_ready(integrate_fn(state))
trajectory = jax.device_get(trajectory)
f0 = to_nodal(trajectory.temperature_variation)
temperature = f0 + reference_temperatures[:, np.newaxis, np.newaxis]
levels = [(220 + 10 * i) for i in range(10)]
plt.contourf(temperature[119, 22, :, :], levels=levels, cmap=plt.cm.Spectral_r)
plt.savefig("b.09.png")
np.asarray(temperature).tofile("b.09.raw")
plt.close()
