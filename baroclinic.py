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


dtype = np.dtype('float32')
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
lon_location = np.pi / 9
lat_location = 2 * np.pi / 9
di.g.longitude_wavenumbers = 22
di.g.total_wavenumbers = 23
di.g.longitude_nodes = 64
di.g.latitude_nodes = 32
di.g.layers = 12
di.g.boundaries = np.linspace(0, 1, di.g.layers + 1, dtype=dtype)
di.g.centers = (di.g.boundaries[1:] + di.g.boundaries[:-1]) / 2
di.g.layer_thickness = np.diff(di.g.boundaries)
di.g.center_to_center = np.diff(di.g.centers)
di.g.f, di.g.p, di.g.w = di.basis()
tau = 0.010938
order = 18
cutoff = 0

longitude = np.linspace(0, 2 * np.pi, di.g.longitude_nodes, endpoint=False)
modal_shape = di.g.layers, 2 * di.g.longitude_wavenumbers - 1, di.g.total_wavenumbers

sin_latitude, _ = scipy.special.roots_legendre(di.g.latitude_nodes)
lon, sin_lat = np.meshgrid(longitude, sin_latitude, indexing="ij")
lat = np.arcsin(sin_lat)
geopotential = np.stack(
    [get_geopotential(lat, sigma) for sigma in di.g.centers])
di.g.reference_temperature = np.stack(
    [get_reference_temperature(sigma) for sigma in di.g.centers])
vorticity = np.stack([get_vorticity(lat, sigma) for sigma in di.g.centers])
orography = get_geopotential(lat, 1.0) / gravity_acceleration
mask = jnp.ones(di.g.total_wavenumbers, dtype).at[-1:].set(0)
di.g.orography = di.transform(jnp.asarray(orography)) * mask
step_fn = di.imex_runge_kutta(di.explicit_terms, di.implicit_terms,
                              di.implicit_inverse, dt)
filter_fn = di.exponential_filter(di.g.total_wavenumbers, dt / tau, order, cutoff)
filters = [di.runge_kutta_step_filter(filter_fn)]
step_fn = di.step_with_filters(step_fn, filters)
vorticity_perturbation = np.stack(
    [get_vorticity_perturbation(lat, lon) for sigma in di.g.centers])
divergence_perturbation = np.stack(
    [get_divergence_perturbation(lat, lon) for sigma in di.g.centers])
temperature_variation = np.stack(
    [get_temperature_variation(lat, sigma) for sigma in di.g.centers])
log_surface_pressure = np.log(p0 * np.ones(lat.shape)[np.newaxis, ...])
state = di.State(
    vorticity=di.transform(jnp.asarray(vorticity)) +
    di.transform(jnp.asarray(vorticity_perturbation)),
    divergence=di.transform(jnp.asarray(divergence_perturbation)),
    temperature_variation=di.transform(jnp.asarray(temperature_variation)),
    log_surface_pressure=di.transform(jnp.asarray(log_surface_pressure)))
final, _ = jax.lax.scan(lambda x, _: (step_fn(x), None),
                        state,
                        xs=None,
                        length=8640)
f0 = di.inverse_transform(final.temperature_variation)
temperature = f0 + di.g.reference_temperature[:, np.newaxis, np.newaxis]
levels = [(220 + 10 * i) for i in range(10)]
plt.contourf(temperature[22, :, :], levels=levels, cmap=plt.cm.Spectral_r)
plt.savefig("b.09.png")
plt.close()
np.asarray(temperature).tofile("b.09.raw")
