import di
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy


def explicit_terms(s):
    l = np.arange(1, di.g.total_wavenumbers)
    inverse_eigenvalues = np.zeros(di.g.total_wavenumbers)
    inverse_eigenvalues[1:] = -1 / (l * (l + 1))
    stream_function = s.vo * inverse_eigenvalues
    velocity_potential = s.di * inverse_eigenvalues
    c00 = di.real_basis_derivative(velocity_potential)
    c01 = di.cos_lat_d_dlat(velocity_potential)
    c10 = di.real_basis_derivative(stream_function)
    c11 = di.cos_lat_d_dlat(stream_function)
    v0 = c00 - c11
    v1 = c01 + c10
    u0 = di.inverse_transform(v0)
    u1 = di.inverse_transform(v1)
    temperature_variation = di.inverse_transform(s.te)
    kv_coeff = kf * (np.maximum(0, (di.g.centers - sigma_b) / (1 - sigma_b)))
    kv = kv_coeff[:, None, None]
    sin_lat, _ = scipy.special.roots_legendre(di.g.latitude_nodes)
    cos2 = 1 - sin_lat**2
    nodal_velocity_tendency = -kv * u0 / cos2, -kv * u1 / cos2
    nodal_temperature = (di.g.temp[:, None, None] + temperature_variation)
    nodal_log_surface_pressure = di.inverse_transform(s.sp)
    nodal_surface_pressure = jnp.exp(nodal_log_surface_pressure)
    p_over_p0 = (di.g.centers[:, None, None] * nodal_surface_pressure / p0)
    temperature = p_over_p0**di.kappa * (maxT - dTy * np.sin(lat)**2 - dThz *
                                         jnp.log(p_over_p0) * np.cos(lat)**2)
    Teq = jnp.maximum(minT, temperature)
    cutoff = np.maximum(0, (di.g.centers - sigma_b) / (1 - sigma_b))
    kt = ka + (ks - ka) * (cutoff[:, None, None] * np.cos(lat)**4)
    u, v = di.transform(jnp.asarray(nodal_velocity_tendency))
    raw_vor = di.real_basis_derivative(v) - di.sec_lat_d_dlat_cos2(u)
    raw_div = di.real_basis_derivative(u) + di.sec_lat_d_dlat_cos2(v)
    mask = np.r_[[1] * (di.g.total_wavenumbers - 1), 0]
    return di.State(raw_vor * mask, raw_div * mask,
                    di.transform(-kt * (nodal_temperature - Teq)),
                    jnp.zeros_like(s.sp))


di.g.longitude_wavenumbers = 22
di.g.total_wavenumbers = 23
di.g.longitude_nodes = 64
di.g.latitude_nodes = 32
di.g.layers = 24
di.g.boundaries = np.linspace(0, 1, di.g.layers + 1, dtype=np.float32)
di.g.centers = (di.g.boundaries[1:] + di.g.boundaries[:-1]) / 2
di.g.thick = np.diff(di.g.boundaries)
di.g.center_to_center = np.diff(di.g.centers)
di.g.f, di.g.p, di.g.w = di.basis()
di.g.temp = np.full((di.g.layers, ), 288)
di.g.geo = di.geopotential_weights()
di.g.tew = di.temperature_weights()

longitude = np.linspace(0, 2 * np.pi, di.g.longitude_nodes, endpoint=False)
sin_latitude, _ = scipy.special.roots_legendre(di.g.latitude_nodes)
lon, sin_lat = np.meshgrid(longitude, sin_latitude, indexing="ij")
lat = np.arcsin(sin_lat)
p0 = 2.9954997684550640e+19
p1 = 1.4977498842275320e+18
nodal_vorticity = jnp.stack([jnp.zeros_like(lat) for sigma in di.g.centers])
modal_vorticity = di.transform(nodal_vorticity)
altitude_m = np.zeros_like(lat)
g = 9.80665
cp = 1004.68506
T0 = 288.16
M = 0.02896968
R0 = 8.314462618
relative_pressure = (1 - g * altitude_m / (cp * T0))**(cp * M / R0)
surface_pressure = p0 * np.ones(
    (1, di.g.longitude_nodes, di.g.latitude_nodes)) * relative_pressure
lon0 = 1.9018228054046631e+00
lat0 = 8.9859783649444580e-02
stddev = np.pi / 20
k = 4
perturbation = (jnp.exp(-((lon - lon0)**2) / (2 * stddev**2)) *
                jnp.exp(-((lat - lat0)**2) /
                        (2 * stddev**2)) * jnp.sin(k * (lon - lon0)))
nodal_surface_pressure = surface_pressure + p1 * perturbation
state = di.State(modal_vorticity, jnp.zeros_like(modal_vorticity),
                 jnp.zeros_like(modal_vorticity),
                 di.transform(jnp.log(nodal_surface_pressure)))
di.g.orography = di.transform(np.zeros_like(lat))
dt = 8.7504000000000012e-02
sigma_b = 0.7
minT = 200
maxT = 315
dTy = 60
dThz = 10
p0 = 2.995499768455064e+19
kf = 7.9361451413014747e-02
ka = 1.9840362853253690e-03
ks = 1.9840362853253687e-02
tau = 0.0087504
order = 1.5
cutoff = 0.8
step_fn = di.runge_kutta(lambda x: di.explicit_terms(x) + explicit_terms(x),
                         di.implicit_terms, di.implicit_inverse, dt)

total_wavenumber = np.arange(di.g.total_wavenumbers)
k = total_wavenumber / total_wavenumber.max()
a = dt / tau
c = cutoff
scaling = jnp.exp((k > c) * (-a * (((k - c) / (1 - c))**(2 * order))))
filter_fn = di._make_filter_fn(scaling)

final, _ = jax.lax.scan(lambda x, _: (filter_fn(step_fn(x)), None),
                        state,
                        xs=None,
                        length=173808)
f0 = di.inverse_transform(final.te)
plt.contourf(f0[22, :, :])
plt.savefig("h.12.png")
np.asarray(f0).tofile("h.12.raw")
plt.close()
