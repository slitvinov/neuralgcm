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


@jax.jit
def vor_div_to_uv_nodal(vorticity, divergence):
    u_cos_lat, v_cos_lat = di.get_cos_lat_vector(vorticity,
                                                 divergence,
                                                 clip=True)
    u_nodal = di.to_nodal(u_cos_lat) / di.cos_lat()
    v_nodal = di.to_nodal(v_cos_lat) / di.cos_lat()
    return u_nodal, v_nodal


def nodal_prognostics_and_diagnostics(state):
    u_nodal, v_nodal = vor_div_to_uv_nodal(state.vorticity, state.divergence)
    surface_geopotential = orography * di.gravity_acceleration
    temperature = state.temperature_variation.at[..., 0, 0].add(
        _CONSTANT_NORMALIZATION_FACTOR * di.g.reference_temperature)
    geopotential_diff = di.get_geopotential_diff(temperature)
    geopotential_nodal = surface_geopotential + geopotential_diff
    vor_nodal = di.to_nodal(state.vorticity)
    div_nodal = di.to_nodal(state.divergence)
    sp_nodal = jnp.exp(di.to_nodal(state.log_surface_pressure))
    tracers_nodal = {k: di.to_nodal(v) for k, v in state.tracers.items()}
    t_nodal = (di.to_nodal(state.temperature_variation) +
               di.g.reference_temperature[:, np.newaxis, np.newaxis])
    sigma_dot_boundaries = di.compute_diagnostic_state(state).sigma_dot_full
    assert sigma_dot_boundaries.ndim == 3
    sigma_dot_padded = jnp.pad(sigma_dot_boundaries, [(1, 1), (0, 0), (0, 0)])
    vertical_velocity_nodal = 0.5 * (sigma_dot_padded[1:] +
                                     sigma_dot_padded[:-1])
    state_nodal = {
        "u_component_of_wind": u_nodal,
        "v_component_of_wind": v_nodal,
        "temperature": t_nodal,
        "vorticity": vor_nodal,
        "divergence": div_nodal,
        "vertical_velocity": vertical_velocity_nodal,
        "geopotential": geopotential_nodal,
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


@jax.jit
def uv_nodal_to_vor_div_modal(u_nodal, v_nodal):
    u_over_cos_lat = di.to_modal(u_nodal / di.cos_lat())
    v_over_cos_lat = di.to_modal(v_nodal / di.cos_lat())
    vorticity = di.curl_cos_lat((u_over_cos_lat, v_over_cos_lat), clip=True)
    divergence = di.div_cos_lat((u_over_cos_lat, v_over_cos_lat), clip=True)
    return vorticity, divergence


@jax.jit
@functools.partial(jax.vmap, in_axes=(-1, None, -1), out_axes=-1)
@functools.partial(jax.vmap, in_axes=(-1, None, -1), out_axes=-1)
def regrid(surface_pressure, target, field):
    source = a_boundaries / surface_pressure + b_boundaries
    upper = jnp.minimum(target[1:, jnp.newaxis], source[jnp.newaxis, 1:])
    lower = jnp.maximum(target[:-1, jnp.newaxis], source[jnp.newaxis, :-1])
    weights = jnp.maximum(upper - lower, 0)
    weights /= jnp.sum(weights, axis=1, keepdims=True)
    return jnp.einsum("ab,b->a", weights, field, precision="float32")


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
    era.to_netcdf("weather.h5")
ds = era[[
    "u_component_of_wind",
    "v_component_of_wind",
    "temperature",
    "specific_humidity",
    "specific_cloud_liquid_water_content",
    "specific_cloud_ice_water_content",
    "surface_pressure",
    "geopotential_at_surface",
]]

ds1 = ds.compute().interp(latitude=desired_lat, longitude=desired_lon)
hyb = era["hybrid"].data
lat = era["latitude"].data
lon = era["longitude"].data
nhyb = len(hyb)
shape = nhyb, len(desired_lon), len(desired_lat)
xi = np.meshgrid(desired_lat, desired_lon)
points = lat, lon
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
    source = era[key].data
    for i in range(nhyb):
        val[i] = scipy.interpolate.interpn(points, source[i], xi)
    M[key] = val / scale
sp_init_hpa = ds1["surface_pressure"].data.T / 100
ds1["orography"] = ds1["geopotential_at_surface"] / (uL * GRAVITY_ACCELERATION)
ds1["surface_pressure"] /= 1 / uL / uT**2
orography_input = ds1["orography"].transpose(..., "longitude",
                                             "latitude").data[np.newaxis, ...]
sp_nodal = ds1["surface_pressure"].transpose(..., "longitude",
                                             "latitude").data[np.newaxis, ...]
nodal_inputs = {
    key: regrid(sp_init_hpa, di.g.boundaries, val)
    for key, val in M.items()
}
u_nodal = nodal_inputs["u_component_of_wind"]
v_nodal = nodal_inputs["v_component_of_wind"]
t_nodal = nodal_inputs["temperature"]
vorticity, divergence = uv_nodal_to_vor_div_modal(u_nodal, v_nodal)
di.g.reference_temperature = 250 * np.ones((di.g.layers, ))
temperature_variation = di.to_modal(
    t_nodal - di.g.reference_temperature.reshape(-1, 1, 1))
log_sp = di.to_modal(np.log(sp_nodal))
tracers = di.to_modal({
    "specific_humidity":
    nodal_inputs["specific_humidity"],
    "specific_cloud_liquid_water_content":
    nodal_inputs["specific_cloud_liquid_water_content"],
    "specific_cloud_ice_water_content":
    nodal_inputs["specific_cloud_ice_water_content"],
})
raw_init_state = di.State(
    vorticity=vorticity,
    divergence=divergence,
    temperature_variation=temperature_variation,
    log_surface_pressure=log_sp,
    tracers=tracers,
)
orography = di.to_modal(orography_input)
orography = di.exponential_filter(di.g.total_wavenumbers, order=2)(orography)
di.g.orography = orography
res_factor = di.g.latitude_nodes / 128
dt = 4.3752000000000006e-02
tau = 3600 * 8.6 / (2.4**np.log2(res_factor)) / uT

eigenvalues = di.laplacian_eigenvalues()
scale = dt / (tau * abs(eigenvalues[-1])**2)
scaling = jnp.exp(-scale * (-eigenvalues)**2)
filter_fn = di._make_filter_fn(scaling)
hyperdiffusion_filter = di.runge_kutta_step_filter(filter_fn)
time_span = cutoff_period = 3.1501440000000001e+00
forward_step = di.step_with_filters(
    di.imex_runge_kutta(di.explicit_terms, di.implicit_terms,
                        di.implicit_inverse, dt), [hyperdiffusion_filter])
backward_step = di.step_with_filters(
    di.imex_runge_kutta(explicit_terms, implicit_terms, implicit_inverse, dt),
    [hyperdiffusion_filter])
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
step_fn = di.step_with_filters(
    di.imex_runge_kutta(di.explicit_terms, di.implicit_terms,
                        di.implicit_inverse, dt),
    [hyperdiffusion_filter],
)


def step_fn0(frame):
    gfun = lambda x, _: (step_fn(x), None)
    x_final, _ = jax.lax.scan(gfun, frame, xs=None, length=inner_steps)
    return x_final


def step(frame, _):
    return step_fn0(frame), nodal_prognostics_and_diagnostics(frame)


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
