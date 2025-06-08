import di
import functools
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

units = di.units


def kv():
    kv_coeff = kf * (np.maximum(0, (sigma - sigma_b) / (1 - sigma_b)))
    return kv_coeff[:, np.newaxis, np.newaxis]


def kt():
    cutoff = np.maximum(0, (sigma - sigma_b) / (1 - sigma_b))
    return ka + (ks - ka) * (cutoff[:, np.newaxis, np.newaxis] *
                             np.cos(lat)**4)


def equilibrium_temperature(nodal_surface_pressure):
    p_over_p0 = (sigma[:, np.newaxis, np.newaxis] * nodal_surface_pressure /
                 p0)
    temperature = p_over_p0**di.kappa * (maxT - dTy * np.sin(lat)**2 - dThz *
                                         jnp.log(p_over_p0) * np.cos(lat)**2)
    return jnp.maximum(minT, temperature)


def explicit_terms(state):
    aux_state = di.compute_diagnostic_state(state=state, coords=coords)
    nodal_velocity_tendency = jax.tree.map(
        lambda x: -kv() * x / coords.horizontal.cos_lat**2,
        aux_state.cos_lat_u,
    )
    nodal_temperature = (ref_temps[:, np.newaxis, np.newaxis] +
                         aux_state.temperature_variation)
    nodal_log_surface_pressure = coords.horizontal.to_nodal(
        state.log_surface_pressure)
    nodal_surface_pressure = jnp.exp(nodal_log_surface_pressure)
    Teq = equilibrium_temperature(nodal_surface_pressure)
    nodal_temperature_tendency = -kt() * (nodal_temperature - Teq)
    temperature_tendency = coords.horizontal.to_modal(
        nodal_temperature_tendency)
    velocity_tendency = coords.horizontal.to_modal(nodal_velocity_tendency)
    vorticity_tendency = coords.horizontal.curl_cos_lat(velocity_tendency)
    divergence_tendency = coords.horizontal.div_cos_lat(velocity_tendency)
    log_surface_pressure_tendency = jnp.zeros_like(state.log_surface_pressure)
    return di.State(
        vorticity=vorticity_tendency,
        divergence=divergence_tendency,
        temperature_variation=temperature_tendency,
        log_surface_pressure=log_surface_pressure_tendency,
    )


def explicit_fn(x):
    return di.tree_map(lambda *args: sum([x for x in args if x is not None]),
                       primitive.explicit_terms(x), explicit_terms(x))


layers = 24
coords = di.CoordinateSystem(horizontal=di.Grid(longitude_wavenumbers=22,
                                                total_wavenumbers=23,
                                                longitude_nodes=64,
                                                latitude_nodes=32),
                             vertical=di.SigmaCoordinates(
                                 np.linspace(0,
                                             1,
                                             layers + 1,
                                             dtype=np.float32)))
p0 = 100e3 * units.pascal
p1 = 5e3 * units.pascal
tref = 288.0 * units.degK
rng_key = jax.random.PRNGKey(0)
lon, sin_lat = coords.horizontal.nodal_mesh
lat = np.arcsin(sin_lat)
tref = di.DEFAULT_SCALE.nondimensionalize(units.Quantity(tref))
p0 = di.DEFAULT_SCALE.nondimensionalize(units.Quantity(p0))
p1 = di.DEFAULT_SCALE.nondimensionalize(units.Quantity(p1))
orography = np.zeros_like(lat)
nodal_vorticity = jnp.stack(
    [jnp.zeros_like(lat) for sigma in coords.vertical.centers])
modal_vorticity = coords.horizontal.to_modal(nodal_vorticity)

altitude_m = di.DEFAULT_SCALE.dimensionalize(orography, units.meter).magnitude
g = 9.80665
cp = 1004.68506
T0 = 288.16
M = 0.02896968
R0 = 8.314462618
relative_pressure = (1 - g * altitude_m / (cp * T0))**(cp * M / R0)
surface_pressure = (p0 * np.ones((1, ) + coords.horizontal.nodal_shape) *
                    relative_pressure)
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
    log_surface_pressure=(coords.horizontal.to_modal(
        jnp.log(nodal_surface_pressure))),
)
ref_temps = np.full((coords.vertical.layers, ), tref)
orography = di.truncated_modal_orography(orography, coords)
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
dt_si = 10 * units.minute
save_every = 10 * units.day
total_time = 1200 * units.day
inner_steps = int(save_every / dt_si)
outer_steps = int(total_time / save_every)
dt = di.DEFAULT_SCALE.nondimensionalize(dt_si)
primitive = di.PrimitiveEquations(ref_temps, orography, coords)
p0 = 1e5 * units.pascal
sigma_b = 0.7
kf = 1 / (1 * units.day)
ka = 1 / (40 * units.day)
ks = 1 / (4 * units.day)
minT = 200 * units.degK
maxT = 315 * units.degK
dTy = 60 * units.degK
dThz = 10 * units.degK
coords = coords
ref_temps = ref_temps
p0 = di.DEFAULT_SCALE.nondimensionalize(p0)
sigma_b = sigma_b
kf = di.DEFAULT_SCALE.nondimensionalize(kf)
ka = di.DEFAULT_SCALE.nondimensionalize(ka)
ks = di.DEFAULT_SCALE.nondimensionalize(ks)
minT = di.DEFAULT_SCALE.nondimensionalize(minT)
maxT = di.DEFAULT_SCALE.nondimensionalize(maxT)
dTy = di.DEFAULT_SCALE.nondimensionalize(dTy)
dThz = di.DEFAULT_SCALE.nondimensionalize(dThz)
sigma = coords.vertical.centers
_, sin_lat = coords.horizontal.nodal_mesh
lat = np.arcsin(sin_lat)

primitive_with_hs = di.ImplicitExplicitODE(explicit_fn,
                                           primitive.implicit_terms,
                                           primitive.implicit_inverse)

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
start_time = 200
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
final, trajectory = jax.block_until_ready(integrate_fn(final))
f0 = di.maybe_to_nodal(trajectory_dict, coords=coords)
plt.contourf(f0["temperature_variation"][-1, 22, :, :])
plt.savefig("h.12.png")
np.asarray(f0["temperature_variation"]).tofile("h.12.raw")
plt.close()
