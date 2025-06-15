import di
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


def equilibrium_temperature(nodal_surface_pressure):
    p_over_p0 = (di.g.centers[:, np.newaxis, np.newaxis] * nodal_surface_pressure /
                 p0)
    temperature = p_over_p0**di.kappa * (maxT - dTy * np.sin(lat)**2 - dThz *
                                         jnp.log(p_over_p0) * np.cos(lat)**2)
    return jnp.maximum(minT, temperature)


def explicit_terms(state):
    aux_state = di.compute_diagnostic_state(state)
    kv_coeff = kf * (np.maximum(0, (di.g.centers - sigma_b) / (1 - sigma_b)))
    kv = kv_coeff[:, np.newaxis, np.newaxis]
    nodal_velocity_tendency = jax.tree.map(
        lambda x: -kv * x / di.cos_lat()**2,
        aux_state.cos_lat_u,
    )
    nodal_temperature = (ref_temps[:, np.newaxis, np.newaxis] +
                         aux_state.temperature_variation)
    nodal_log_surface_pressure = di.inverse_transform(
        state.log_surface_pressure)
    nodal_surface_pressure = jnp.exp(nodal_log_surface_pressure)
    Teq = equilibrium_temperature(nodal_surface_pressure)
    cutoff = np.maximum(0, (di.g.centers - sigma_b) / (1 - sigma_b))
    kt = ka + (ks - ka) * (cutoff[:, np.newaxis, np.newaxis] * np.cos(lat)**4)
    nodal_temperature_tendency = -kt * (nodal_temperature - Teq)
    temperature_tendency = di.to_modal(nodal_temperature_tendency)
    velocity_tendency = di.to_modal(nodal_velocity_tendency)
    vorticity_tendency = di.curl_cos_lat(velocity_tendency)
    divergence_tendency = di.div_cos_lat(velocity_tendency)
    log_surface_pressure_tendency = jnp.zeros_like(state.log_surface_pressure)
    return di.State(
        vorticity=vorticity_tendency,
        divergence=divergence_tendency,
        temperature_variation=temperature_tendency,
        log_surface_pressure=log_surface_pressure_tendency,
    )


def explicit_fn(x):
    return di.tree_map(lambda *args: sum([x for x in args if x is not None]),
                       di.explicit_terms(x), explicit_terms(x))


di.g.longitude_wavenumbers = 22
di.g.total_wavenumbers = 23
di.g.longitude_nodes = 64
di.g.latitude_nodes = 32
di.g.layers = 24
di.g.boundaries = np.linspace(0, 1, di.g.layers + 1, dtype=np.float32)
di.g.centers = (di.g.boundaries[1:] + di.g.boundaries[:-1]) / 2
di.g.layer_thickness = np.diff(di.g.boundaries)
di.g.center_to_center = np.diff(di.g.centers)
di.g.f, di.g.p, di.g.w = di.basis()

tref = 288.0
rng_key = jax.random.PRNGKey(0)
lon, sin_lat = di.nodal_mesh()
lat = np.arcsin(sin_lat)
p0 = 2.9954997684550640e+19
p1 = 1.4977498842275320e+18
orography = np.zeros_like(lat)
nodal_vorticity = jnp.stack([jnp.zeros_like(lat) for sigma in di.g.centers])
modal_vorticity = di.to_modal(nodal_vorticity)
altitude_m = np.zeros_like(lat)
g = 9.80665
cp = 1004.68506
T0 = 288.16
M = 0.02896968
R0 = 8.314462618
relative_pressure = (1 - g * altitude_m / (cp * T0))**(cp * M / R0)
surface_pressure = p0 * np.ones(
    (1, di.g.longitude_nodes, di.g.latitude_nodes)) * relative_pressure
keys = jax.random.split(rng_key, 2)
lon0 = jax.random.uniform(keys[1], minval=np.pi / 2, maxval=3 * np.pi / 2)
lat0 = jax.random.uniform(keys[0], minval=-np.pi / 4, maxval=np.pi / 4)
stddev = np.pi / 20
k = 4
perturbation = (jnp.exp(-((lon - lon0)**2) / (2 * stddev**2)) *
                jnp.exp(-((lat - lat0)**2) /
                        (2 * stddev**2)) * jnp.sin(k * (lon - lon0)))
nodal_surface_pressure = surface_pressure + p1 * perturbation

initial_state = di.State(
    vorticity=modal_vorticity,
    divergence=jnp.zeros_like(modal_vorticity),
    temperature_variation=jnp.zeros_like(modal_vorticity),
    log_surface_pressure=(di.to_modal(jnp.log(nodal_surface_pressure))),
)
ref_temps = np.full((di.g.layers, ), tref)
orography = di.clip_wavenumbers(di.to_modal(orography))

inner_steps = 2
outer_steps = 144
dt = 4.3752000000000006e-02
di.g.reference_temperature = ref_temps
di.g.orography = orography
step_fn = di.imex_runge_kutta(di.explicit_terms, di.implicit_terms,
                              di.implicit_inverse, dt)
filters = [di.exponential_step_filter(di.g.total_wavenumbers, dt)]
step_fn = di.step_with_filters(step_fn, filters)
integrate_fn = jax.jit(
    di.trajectory_from_step(step_fn,
                            outer_steps=outer_steps,
                            inner_steps=inner_steps,
                            start_with_input=True))
final, trajectory = integrate_fn(initial_state)
inner_steps = 1440
outer_steps = 120
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
_, sin_lat = di.nodal_mesh()
lat = np.arcsin(sin_lat)
step_fn = di.imex_runge_kutta(explicit_fn, di.implicit_terms,
                              di.implicit_inverse, dt)
filters = [
    di.exponential_step_filter(di.g.total_wavenumbers,
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
final, trajectory = integrate_fn(initial_state)
start_time = 200
inner_steps = 36
outer_steps = 28
dt = 8.7504000000000012e-02
step_fn = di.imex_runge_kutta(explicit_fn, di.implicit_terms,
                              di.implicit_inverse, dt)
filters = [
    di.exponential_step_filter(di.g.total_wavenumbers,
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
final, trajectory = integrate_fn(final)
f0 = di.inverse_transform(trajectory.temperature_variation)
plt.contourf(f0[-1, 22, :, :])
plt.savefig("h.12.png")
np.asarray(f0).tofile("h.12.raw")
plt.close()
