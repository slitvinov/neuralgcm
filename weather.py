import dataclasses
import di
import functools
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pint
import xarray

units = pint.UnitRegistry(autoconvert_offset_to_baseunit=True)
Unit = units.Unit
GRAVITY_ACCELERATION = 9.80616 * units.m / units.s**2


class Scale:

    def __init__(self, *scales):
        self.scales = {}
        for quantity in scales:
            self.scales[str(
                quantity.dimensionality)] = quantity.to_base_units()

    def scaling_factor(self, dimensionality):
        factor = units.Quantity(1)
        for dimension, exponent in dimensionality.items():
            quantity = self.scales.get(dimension)
            factor *= quantity**exponent
        assert factor.check(dimensionality)
        return factor

    def nondimensionalize(self, quantity):
        scaling_factor = self.scaling_factor(quantity.dimensionality)
        nondimensionalized = (quantity / scaling_factor).to(
            units.dimensionless)
        return nondimensionalized.magnitude

    def dimensionalize(self, value, unit):
        scaling_factor = self.scaling_factor(unit.dimensionality)
        dimensionalized = value * scaling_factor
        return dimensionalized.to(unit)


DEFAULT_SCALE = Scale(
    6.37122e6 * units.m,
    1 / 2 / 7.292e-5 * units.s,
    1 * units.kilogram,
    1 * units.degK,
)


def attach_data_array_units(array):
    attrs = dict(array.attrs)
    units0 = attrs.pop("units", None)
    data = units.parse_expression(units0) * array.data
    return xarray.DataArray(data, array.coords, array.dims, attrs=attrs)


def xarray_nondimensionalize(ds):
    return xarray.apply_ufunc(DEFAULT_SCALE.nondimensionalize, ds)


def xarray_to_gcm_dict(ds):
    var_names = ds.keys()
    result = {}
    for var_name in var_names:
        data = ds[var_name].transpose(..., "longitude", "latitude").data
        if data.ndim == 2:
            data = data[np.newaxis, ...]
        result[var_name] = data
    return result


def slice_levels(output, level_indices):

    def get_horizontal(x):
        if x.shape[0] == 1:
            return x
        else:
            return x[level_indices, ...]

    return jax.tree.map(get_horizontal, output)


def open_era5(path, time):
    ds = xarray.open_zarr(path,
                          chunks=None,
                          storage_options=dict(token="anon"))
    return ds.sel(time=time)


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
    geopotential_nodal = di.to_nodal(
        di.get_geopotential(
            state.temperature_variation,
            di.g.reference_temperature,
            orography,
        ))
    vor_nodal = di.to_nodal(state.vorticity)
    div_nodal = di.to_nodal(state.divergence)
    sp_nodal = jnp.exp(di.to_nodal(state.log_surface_pressure))
    tracers_nodal = {k: di.to_nodal(v) for k, v in state.tracers.items()}
    t_nodal = (di.to_nodal(state.temperature_variation) +
               di.g.reference_temperature[:, np.newaxis, np.newaxis])
    vertical_velocity_nodal = compute_vertical_velocity(state)
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
    return slice_levels(state_nodal, output_level_indices)


def trajectory_to_xarray(trajectory):
    target_units = {k: v.data.units for k, v in ds_init.items()}
    target_units |= {
        "vorticity": units("1/s"),
        "divergence": units("1/s"),
        "geopotential": units("m^2/s^2"),
        "vertical_velocity": units("1/s"),
    }
    orography_nodal = jax.device_put(di.to_nodal(orography),
                                     device=jax.devices("cpu")[0])
    trajectory_cpu = jax.device_put(trajectory, device=jax.devices("cpu")[0])
    traj_nodal_si = {
        k: DEFAULT_SCALE.dimensionalize(v, target_units[k]).magnitude
        for k, v in trajectory_cpu.items()
    }
    times = float(save_every / units.hour) * np.arange(outer_steps)
    lon = 180 / np.pi * di.nodal_axes()[0]
    lat = 180 / np.pi * np.arcsin(di.nodal_axes()[1])
    dims = ("time", "sigma", "longitude", "latitude")
    ds_result = xarray.Dataset(
        data_vars={
            k: (dims, v)
            for k, v in traj_nodal_si.items() if k != "surface_pressure"
        },
        coords={
            "longitude": lon,
            "latitude": lat,
            "sigma": di.g.centers[output_level_indices],
            "time": times,
            "orography":
            (("longitude", "latitude"), orography_nodal.squeeze()),
        },
    ).assign(surface_pressure=(
        ("time", "longitude", "latitude"),
        traj_nodal_si["surface_pressure"].squeeze(axis=-3),
    ))
    return ds_result


def conservative_regrid_weights(source, target):
    upper = jnp.minimum(target[1:, jnp.newaxis], source[jnp.newaxis, 1:])
    lower = jnp.maximum(target[:-1, jnp.newaxis], source[jnp.newaxis, :-1])
    weights = jnp.maximum(upper - lower, 0)
    weights /= jnp.sum(weights, axis=1, keepdims=True)
    assert weights.shape == (target.size - 1, source.size - 1)
    return weights


@jax.jit
def regrid_hybrid_to_sigma(fields, surface_pressure):

    @jax.jit
    @functools.partial(jnp.vectorize, signature="(x,y),(a),(b,x,y)->(c,x,y)")
    @functools.partial(jax.vmap, in_axes=(-1, None, -1), out_axes=-1)
    @functools.partial(jax.vmap, in_axes=(-1, None, -1), out_axes=-1)
    def regrid(surface_pressure, sigma_bounds, field):
        assert sigma_bounds.shape == (di.g.layers + 1, )
        hybrid_bounds = a_boundaries / surface_pressure + b_boundaries
        weights = conservative_regrid_weights(hybrid_bounds, sigma_bounds)
        result = jnp.einsum("ab,b->a", weights, field, precision="float32")
        assert result.shape[0] == di.g.layers
        return result

    return di.tree_map_over_nonscalars(
        lambda x: regrid(surface_pressure, di.g.boundaries, x), fields)


def horizontal_diffusion_step_filter(dt, tau, order=1):
    eigenvalues = di.laplacian_eigenvalues()
    scale = dt / (tau * abs(eigenvalues[-1])**order)
    filter_fn = horizontal_diffusion_filter(scale, order)
    return di.runge_kutta_step_filter(filter_fn)


def horizontal_diffusion_filter(scale, order=1):
    eigenvalues = di.laplacian_eigenvalues()
    scaling = jnp.exp(-scale * (-eigenvalues)**order)
    return di._make_filter_fn(scaling)


def compute_vertical_velocity(state):
    sigma_dot_boundaries = di.compute_diagnostic_state(state).sigma_dot_full
    assert sigma_dot_boundaries.ndim == 3
    sigma_dot_padded = jnp.pad(sigma_dot_boundaries, [(1, 1), (0, 0), (0, 0)])
    return 0.5 * (sigma_dot_padded[1:] + sigma_dot_padded[:-1])


class TimeReversedImExODE:

    def explicit_terms(self, state):
        forward_term = di.explicit_terms(state)
        return di.tree_map(jnp.negative, forward_term)

    def implicit_terms(self, state):
        forward_term = di.implicit_terms(state)
        return di.tree_map(jnp.negative, forward_term)

    def implicit_inverse(self, state, step_size):
        return di.implicit_inverse(state, -step_size)


def accumulate_repeated(step_fn, weights, state, scan_fn=jax.lax.scan):

    def f(carry, weight):
        state, averaged = carry
        state = step_fn(state)
        averaged = di.tree_map(lambda s, a: a + weight * s, state, averaged)
        return (state, averaged), None

    zeros = di.tree_map(jnp.zeros_like, state)
    init = (state, zeros)
    (_, averaged), _ = scan_fn(f, init, weights)
    return averaged


@jax.jit
def uv_nodal_to_vor_div_modal(u_nodal, v_nodal):
    u_over_cos_lat = di.to_modal(u_nodal / di.cos_lat())
    v_over_cos_lat = di.to_modal(v_nodal / di.cos_lat())
    vorticity = di.curl_cos_lat((u_over_cos_lat, v_over_cos_lat), clip=True)
    divergence = di.div_cos_lat((u_over_cos_lat, v_over_cos_lat), clip=True)
    return vorticity, divergence


ref_temp_si = 250
di.g.longitude_wavenumbers = 171
di.g.total_wavenumbers = 172
di.g.longitude_nodes = 512
di.g.latitude_nodes = 256
di.g.layers = 32
di.g.boundaries = np.linspace(0, 1, di.g.layers + 1, dtype=np.float32)
di.g.centers = (di.g.boundaries[1:] + di.g.boundaries[:-1]) / 2
di.g.layer_thickness = np.diff(di.g.boundaries)
di.g.center_to_center = np.diff(di.g.centers)
dt_si = 5 * units.minute
save_every = 15 * units.minute
total_time = 2 * units.day + save_every
dfi_timescale = 6 * units.hour
output_level_indices = [
    di.g.layers // 4, di.g.layers // 2, 3 * di.g.layers // 4, -1
]

ds_arco_era5 = xarray.merge([
    open_era5(
        "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
        time="19900501T00",
    ).drop_dims("level"),
    open_era5(
        "gs://gcp-public-data-arco-era5/ar/model-level-1h-0p25deg.zarr-v1",
        time="19900501T00",
    ),
])
ds = ds_arco_era5[[
    "u_component_of_wind",
    "v_component_of_wind",
    "temperature",
    "specific_humidity",
    "specific_cloud_liquid_water_content",
    "specific_cloud_ice_water_content",
    "surface_pressure",
]]
raw_orography = ds_arco_era5.geopotential_at_surface
desired_lon = 180 / np.pi * di.nodal_axes()[0]
desired_lat = 180 / np.pi * np.arcsin(di.nodal_axes()[1])
ds0 = ds.compute().interp(latitude=desired_lat, longitude=desired_lon)
ds_init = ds0.map(attach_data_array_units)
ds_init["orography"] = attach_data_array_units(
    raw_orography.interp(latitude=desired_lat, longitude=desired_lon))
ds_init["orography"] /= GRAVITY_ACCELERATION

a_in_pa, b_boundaries = np.loadtxt("ecmwf137_hybrid_levels.csv",
                                   skiprows=1,
                                   usecols=(1, 2),
                                   delimiter="\t").T
a_boundaries = a_in_pa / 100
ds_nondim_init = xarray_nondimensionalize(ds_init)
model_level_inputs = xarray_to_gcm_dict(ds_nondim_init)
sp_nodal = model_level_inputs.pop("surface_pressure")
orography_input = model_level_inputs.pop("orography")
sp_init_hpa = (ds_init.surface_pressure.transpose(
    "longitude", "latitude").data.to("hPa").magnitude)
nodal_inputs = regrid_hybrid_to_sigma(model_level_inputs, sp_init_hpa)
u_nodal = nodal_inputs["u_component_of_wind"]
v_nodal = nodal_inputs["v_component_of_wind"]
t_nodal = nodal_inputs["temperature"]
vorticity, divergence = uv_nodal_to_vor_div_modal(u_nodal, v_nodal)
di.g.reference_temperature = ref_temp_si * np.ones((di.g.layers, ))
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
dt = DEFAULT_SCALE.nondimensionalize(dt_si)
tau = DEFAULT_SCALE.nondimensionalize(8.6 / (2.4**np.log2(res_factor)) *
                                      units.hours)
hyperdiffusion_filter = horizontal_diffusion_step_filter(dt=dt,
                                                         tau=tau,
                                                         order=2)
time_span = cutoff_period = DEFAULT_SCALE.nondimensionalize(dfi_timescale)


def fun(state):
    forward_step = di.step_with_filters(
        di.imex_runge_kutta0(di.explicit_terms, di.implicit_terms,
                             di.implicit_inverse, dt), [hyperdiffusion_filter])
    teq = TimeReversedImExODE()
    backward_step = di.step_with_filters(di.imex_runge_kutta(teq, dt),
                                         [hyperdiffusion_filter])
    N = round(time_span / (2 * dt))
    n = np.arange(1, N + 1)
    weights = np.sinc(n / (N + 1)) * np.sinc(n * time_span /
                                             (cutoff_period * N))
    init_weight = 1.0
    total_weight = init_weight + 2 * weights.sum()
    init_weight /= total_weight
    weights /= total_weight
    init_term = di.tree_map(lambda x: x * init_weight, state)
    forward_term = accumulate_repeated(forward_step, weights, state)
    backward_term = accumulate_repeated(backward_step, weights, state)
    return di.tree_map(lambda *xs: sum(xs), init_term, forward_term,
                       backward_term)


dfi = jax.jit(fun)
dfi_init_state = jax.block_until_ready(dfi(raw_init_state))

inner_steps = int(save_every / dt_si)
outer_steps = int(total_time / save_every)
step_fn = di.step_with_filters(
    di.imex_runge_kutta0(di.explicit_terms, di.implicit_terms,
                         di.implicit_inverse, dt),
    [hyperdiffusion_filter],
)
integrate_fn = jax.jit(
    di.trajectory_from_step(
        step_fn,
        outer_steps=outer_steps,
        inner_steps=inner_steps,
        start_with_input=True,
        post_process_fn=nodal_prognostics_and_diagnostics,
    ))
out_state, trajectory = jax.block_until_ready(integrate_fn(dfi_init_state))
ds_out = trajectory_to_xarray(trajectory)
out_state, trajectory = jax.block_until_ready(integrate_fn(raw_init_state))
ds_out_unfiltered = trajectory_to_xarray(trajectory)
ds_out.surface_pressure.sel(
    latitude=0, longitude=0,
    method="nearest").plot.line(label="digital filter initialization")
ds_out_unfiltered.surface_pressure.sel(
    latitude=0, longitude=0, method="nearest").plot.line(label="unfiltered")
plt.legend()
plt.savefig("w.00.png")
np.asarray(ds_out.surface_pressure.data).tofile("w.00.raw")
plt.close()
ds_out.specific_humidity.thin(time=4 * 24).isel(sigma=1).plot.imshow(
    col="time",
    x="longitude",
    y="latitude",
    col_wrap=3,
    aspect=2,
    size=3.5,
    cmap="viridis",
    vmin=0,
    vmax=0.01,
)
plt.savefig("w.01.png")
np.asarray(ds_out.specific_humidity.data).tofile("w.01.raw")
plt.close()
ds_out.specific_cloud_liquid_water_content.thin(time=4 *
                                                24).isel(sigma=2).plot.imshow(
                                                    col="time",
                                                    x="longitude",
                                                    y="latitude",
                                                    col_wrap=3,
                                                    aspect=2,
                                                    size=3.5,
                                                    cmap="RdBu",
                                                    vmin=-1e-4,
                                                    vmax=1e-4,
                                                )
plt.savefig("w.02.png")
np.asarray(ds_out.specific_cloud_liquid_water_content.data).tofile("w.02.raw")
plt.close()
