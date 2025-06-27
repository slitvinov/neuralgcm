import dataclasses
import di
import functools
import jax
import jax.numpy as jnp
import numpy as np
import os
import scipy
import xarray

GRAVITY_ACCELERATION = 9.80616  #  * units.m / units.s**2
_CONSTANT_NORMALIZATION_FACTOR = 3.5449077

uL = 6.37122e6
uT = 1 / 2 / 7.292e-5


def open(path):
    x = xarray.open_zarr(path, chunks=None, storage_options=dict(token="anon"))
    return x.sel(time="19900501T00")


def nodal_prognostics_and_diagnostics(state):
    sp_nodal = jnp.exp(di.to_nodal(state.log_surface_pressure))
    tracers_nodal = {k: di.to_nodal(v) for k, v in state.tracers.items()}
    state_nodal = {
        "surface_pressure": sp_nodal,
        **tracers_nodal,
    }

    def get_horizontal(x):
        if x.shape[0] == 1:
            return x
        else:
            return x[output_level_indices, ...]

    return jax.tree.map(get_horizontal, state_nodal)


def explicit_terms(state):
    forward_term = di.explicit_terms(state)
    return di.tree_map(jnp.negative, forward_term)


def implicit_terms(state):
    forward_term = di.implicit_terms(state)
    return di.tree_map(jnp.negative, forward_term)


def implicit_inverse(state, step_size):
    return di.implicit_inverse(state, -step_size)


def accumulate_repeated(step_fn, weights, state):

    def f(carry, weight):
        state, averaged = carry
        state = step_fn(state)
        averaged = di.tree_map(lambda s, a: a + weight * s, state, averaged)
        return (state, averaged), None

    zeros = di.tree_map(jnp.zeros_like, state)
    init = (state, zeros)
    (_, averaged), _ = jax.lax.scan(f, init, weights)
    return averaged


def cos_lat():
    sin_lat, _ = scipy.special.roots_legendre(g.latitude_nodes)
    return np.sqrt(1 - sin_lat**2)


@jax.jit
def uv_nodal_to_vor_div_modal(u_nodal, v_nodal):
    u_over_cos_lat = di.to_modal(u_nodal / cos_lat())
    v_over_cos_lat = di.to_modal(v_nodal / cos_lat())
    vorticity = di.curl_cos_lat((u_over_cos_lat, v_over_cos_lat), clip=True)
    divergence = di.div_cos_lat((u_over_cos_lat, v_over_cos_lat), clip=True)
    return vorticity, divergence


@functools.partial(jax.vmap, in_axes=(-1, None, -1), out_axes=-1)
@functools.partial(jax.vmap, in_axes=(-1, None, -1), out_axes=-1)
def regrid(surface_pressure, target, field):
    source = a_boundaries / surface_pressure + b_boundaries
    upper = jnp.minimum(target[1:, jnp.newaxis], source[jnp.newaxis, 1:])
    lower = jnp.maximum(target[:-1, jnp.newaxis], source[jnp.newaxis, :-1])
    weights = jnp.maximum(upper - lower, 0)
    weights /= jnp.sum(weights, axis=1, keepdims=True)
    return jnp.einsum("ab,b->a", weights, field, precision="float32")


def step_with_filters(fun):

    def step(u):
        return hyperdiffusion_filter(fun(u))

    return step


di.g.longitude_wavenumbers = 171
di.g.total_wavenumbers = 172
di.g.longitude_nodes = 512
di.g.latitude_nodes = 256
di.g.layers = 32
di.g.boundaries = np.linspace(0, 1, di.g.layers + 1, dtype=np.float32)
di.g.centers = (di.g.boundaries[1:] + di.g.boundaries[:-1]) / 2
di.g.layer_thickness = np.diff(di.g.boundaries)
di.g.center_to_center = np.diff(di.g.centers)
di.g.f, di.g.p, di.g.w = di.basis()
output_level_indices = [
    di.g.layers // 4, di.g.layers // 2, 3 * di.g.layers // 4, -1
]
sin_latitude, _ = scipy.special.roots_legendre(di.g.latitude_nodes)
desired_lat = np.rad2deg(np.arcsin(sin_latitude))
desired_lon = np.linspace(0, 360, di.g.longitude_nodes, endpoint=False)
a_in_pa, b_boundaries = np.loadtxt("ecmwf137_hybrid_levels.csv",
                                   skiprows=1,
                                   usecols=(1, 2),
                                   delimiter="\t").T
a_boundaries = a_in_pa / 100
if os.path.exists("weather.h5"):
    era = xarray.open_dataset("weather.h5")
else:
    era = xarray.merge([
        open(
            "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
        ).drop_dims("level"),
        open(
            "gs://gcp-public-data-arco-era5/ar/model-level-1h-0p25deg.zarr-v1")
    ])
    era = era[[
        "u_component_of_wind",
        "v_component_of_wind",
        "temperature",
        "specific_humidity",
        "specific_cloud_liquid_water_content",
        "specific_cloud_ice_water_content",
        "surface_pressure",
        "geopotential_at_surface",
    ]]
    era.to_netcdf("weather.h5")
hyb = era["hybrid"].data
lat = era["latitude"].data
lon = era["longitude"].data
nhyb = len(hyb)
shape = nhyb, len(desired_lon), len(desired_lat)
xi = np.meshgrid(desired_lat, desired_lon)
points = lat, lon
sp = scipy.interpolate.interpn(points, era["surface_pressure"].data, xi)
oro = scipy.interpolate.interpn(points, era["geopotential_at_surface"].data,
                                xi)
sp_init_hpa = sp / 100
sp_nodal = sp[np.newaxis, ...] / (1 / uL / uT**2)
orography_input = oro[np.newaxis, ...] / (uL * GRAVITY_ACCELERATION)
M = {}
for key, scale in [
    ("u_component_of_wind", uL / uT),
    ("v_component_of_wind", uL / uT),
    ("temperature", 1),
    ("specific_cloud_liquid_water_content", 1),
    ("specific_cloud_ice_water_content", 1),
    ("specific_humidity", 1),
]:
    val = np.empty(shape)
    for i in range(nhyb):
        val[i] = scipy.interpolate.interpn(points, era[key].data[i], xi)
    M[key] = regrid(sp_init_hpa, di.g.boundaries, val / scale)
u_nodal = M["u_component_of_wind"]
v_nodal = M["v_component_of_wind"]
t_nodal = M["temperature"]
vorticity, divergence = uv_nodal_to_vor_div_modal(u_nodal, v_nodal)
di.g.reference_temperature = np.full((di.g.layers, ), 250)
temperature_variation = di.transform(
    t_nodal - di.g.reference_temperature.reshape(-1, 1, 1))
log_sp = di.to_modal(np.log(sp_nodal))
tracers = di.to_modal({
    "specific_humidity":
    M["specific_humidity"],
    "specific_cloud_liquid_water_content":
    M["specific_cloud_liquid_water_content"],
    "specific_cloud_ice_water_content":
    M["specific_cloud_ice_water_content"],
})
raw_init_state = di.State(vorticity, divergence, temperature_variation, log_sp,
                          tracers)
total_wavenumber = np.arange(di.g.total_wavenumbers)
k = total_wavenumber / total_wavenumber.max()
orography = di.to_modal(orography_input) * jnp.exp((k > 0) * (-16) * k**4)
di.g.orography = orography
res_factor = di.g.latitude_nodes / 128
dt = 4.3752000000000006e-02
tau = 3600 * 8.6 / (2.4**np.log2(res_factor)) / uT

eigenvalues = di.laplacian_eigenvalues()
scale = dt / (tau * abs(eigenvalues[-1])**2)
scaling = jnp.exp(-scale * (-eigenvalues)**2)
hyperdiffusion_filter = di._make_filter_fn(scaling)
time_span = cutoff_period = 3.1501440000000001e+00
forward_step = step_with_filters(
    di.imex_runge_kutta(di.explicit_terms, di.implicit_terms,
                        di.implicit_inverse, dt))
backward_step = step_with_filters(
    di.imex_runge_kutta(explicit_terms, implicit_terms, implicit_inverse, dt))
N = round(time_span / (2 * dt))
n = np.arange(1, N + 1)
weights = np.sinc(n / (N + 1)) * np.sinc(n * time_span / (cutoff_period * N))
init_weight = 1.0
total_weight = init_weight + 2 * weights.sum()
init_weight /= total_weight
weights /= total_weight
init_term = di.tree_map(lambda x: x * init_weight, raw_init_state)
forward_term = accumulate_repeated(forward_step, weights, raw_init_state)
backward_term = accumulate_repeated(backward_step, weights, raw_init_state)
dfi_init_state = di.tree_map(lambda *xs: sum(xs), init_term, forward_term,
                             backward_term)

inner_steps = 3
outer_steps = 193
times = 0.25 * np.arange(outer_steps)
step_fn = step_with_filters(
    di.imex_runge_kutta(di.explicit_terms, di.implicit_terms,
                        di.implicit_inverse, dt))


def step(frame, _):
    gfun = lambda x, _: (step_fn(x), None)
    x_final, _ = jax.lax.scan(gfun, frame, xs=None, length=inner_steps)
    return x_final, nodal_prognostics_and_diagnostics(frame)


out_state, trajectory0 = jax.lax.scan(step,
                                      dfi_init_state,
                                      xs=None,
                                      length=outer_steps)
out_state, trajectory = jax.lax.scan(step,
                                     raw_init_state,
                                     xs=None,
                                     length=outer_steps)
np.asarray(trajectory["surface_pressure"]).tofile("w.00.raw")
np.asarray(trajectory0["specific_humidity"]).tofile("w.01.raw")
np.asarray(
    trajectory0["specific_cloud_liquid_water_content"]).tofile("w.02.raw")
