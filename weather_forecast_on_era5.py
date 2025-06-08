import di
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import dataclasses
import xarray
import functools

units = di.units


class HybridCoordinates:

    def __init__(self):
        a_in_pa, self.b_boundaries = np.loadtxt("ecmwf137_hybrid_levels.csv",
                                                skiprows=1,
                                                usecols=(1, 2),
                                                delimiter="\t").T
        self.a_boundaries = a_in_pa / 100

    def __hash__(self):
        return hash((tuple(self.a_boundaries.tolist()),
                     tuple(self.b_boundaries.tolist())))

    def get_sigma_boundaries(self, surface_pressure):
        return self.a_boundaries / surface_pressure + self.b_boundaries


def attach_data_array_units(array):
    attrs = dict(array.attrs)
    units = attrs.pop("units", None)
    data = di.units.parse_expression(units) * array.data
    return xarray.DataArray(data, array.coords, array.dims, attrs=attrs)


def attach_xarray_units(ds):
    return ds.map(attach_data_array_units)


def xarray_nondimensionalize(ds):
    return xarray.apply_ufunc(di.DEFAULT_SCALE.nondimensionalize, ds)


def xarray_to_gcm_dict(ds, var_names=None):
    if var_names is None:
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


def nodal_prognostics_and_diagnostics(state):
    coords = model_coords.horizontal
    u_nodal, v_nodal = di.vor_div_to_uv_nodal(coords, state.vorticity,
                                              state.divergence)
    geopotential_nodal = coords.to_nodal(
        di.get_geopotential(
            state.temperature_variation,
            eq.reference_temperature,
            orography,
            model_coords.vertical,
        ))
    vor_nodal = coords.to_nodal(state.vorticity)
    div_nodal = coords.to_nodal(state.divergence)
    sp_nodal = jnp.exp(coords.to_nodal(state.log_surface_pressure))
    tracers_nodal = {k: coords.to_nodal(v) for k, v in state.tracers.items()}
    t_nodal = (coords.to_nodal(state.temperature_variation) +
               ref_temps[:, np.newaxis, np.newaxis])
    vertical_velocity_nodal = di.compute_vertical_velocity(state, model_coords)
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
    orography_nodal = jax.device_put(
        model_coords.horizontal.to_nodal(orography),
        device=jax.devices("cpu")[0])
    trajectory_cpu = jax.device_put(trajectory, device=jax.devices("cpu")[0])
    traj_nodal_si = {
        k: di.DEFAULT_SCALE.dimensionalize(v, target_units[k]).magnitude
        for k, v in trajectory_cpu.items()
    }
    times = float(save_every / units.hour) * np.arange(outer_steps)
    lon = 180 / np.pi * model_coords.horizontal.nodal_axes[0]
    lat = 180 / np.pi * np.arcsin(model_coords.horizontal.nodal_axes[1])
    dims = ("time", "sigma", "longitude", "latitude")
    ds_result = xarray.Dataset(
        data_vars={
            k: (dims, v)
            for k, v in traj_nodal_si.items() if k != "surface_pressure"
        },
        coords={
            "longitude": lon,
            "latitude": lat,
            "sigma": model_coords.vertical.centers[output_level_indices],
            "time": times,
            "orography":
            (("longitude", "latitude"), orography_nodal.squeeze()),
        },
    ).assign(surface_pressure=(
        ("time", "longitude", "latitude"),
        traj_nodal_si["surface_pressure"].squeeze(axis=-3),
    ))
    return ds_result


def _interval_overlap(source_bounds, target_bounds):
    upper = jnp.minimum(target_bounds[1:, jnp.newaxis],
                        source_bounds[jnp.newaxis, 1:])
    lower = jnp.maximum(target_bounds[:-1, jnp.newaxis],
                        source_bounds[jnp.newaxis, :-1])
    return jnp.maximum(upper - lower, 0)


def conservative_regrid_weights(source_bounds, target_bounds):
    weights = _interval_overlap(source_bounds, target_bounds)
    weights /= jnp.sum(weights, axis=1, keepdims=True)
    assert weights.shape == (target_bounds.size - 1, source_bounds.size - 1)
    return weights


@functools.partial(jax.jit, static_argnums=(1, 2))
def regrid_hybrid_to_sigma(fields, hybrid_coords, sigma_coords,
                           surface_pressure):

    @jax.jit
    @functools.partial(jnp.vectorize, signature="(x,y),(a),(b,x,y)->(c,x,y)")
    @functools.partial(jax.vmap, in_axes=(-1, None, -1), out_axes=-1)
    @functools.partial(jax.vmap, in_axes=(-1, None, -1), out_axes=-1)
    def regrid(surface_pressure, sigma_bounds, field):
        assert sigma_bounds.shape == (sigma_coords.layers + 1, )
        hybrid_bounds = hybrid_coords.get_sigma_boundaries(surface_pressure)
        weights = conservative_regrid_weights(hybrid_bounds, sigma_bounds)
        result = jnp.einsum("ab,b->a", weights, field, precision="float32")
        assert result.shape[0] == sigma_coords.layers
        return result

    return di.tree_map_over_nonscalars(
        lambda x: regrid(surface_pressure, sigma_coords.boundaries, x), fields)


def horizontal_diffusion_step_filter(grid, dt, tau, order=1):
    eigenvalues = grid.laplacian_eigenvalues
    scale = dt / (tau * abs(eigenvalues[-1])**order)
    filter_fn = horizontal_diffusion_filter(grid, scale, order)
    return di.runge_kutta_step_filter(filter_fn)


def horizontal_diffusion_filter(grid, scale, order=1):
    eigenvalues = grid.laplacian_eigenvalues
    scaling = jnp.exp(-scale * (-eigenvalues)**order)
    return di._make_filter_fn(scaling)


def compute_vertical_velocity(state, coords):
    sigma_dot_boundaries = di.compute_diagnostic_state(state,
                                                       coords).sigma_dot_full
    assert sigma_dot_boundaries.ndim == 3
    sigma_dot_padded = jnp.pad(sigma_dot_boundaries, [(1, 1), (0, 0), (0, 0)])
    return 0.5 * (sigma_dot_padded[1:] + sigma_dot_padded[:-1])


@functools.partial(jax.jit, static_argnames=("grid", "clip"))
def uv_nodal_to_vor_div_modal(grid, u_nodal, v_nodal, clip=True):
    u_over_cos_lat = grid.to_modal(u_nodal / grid.cos_lat)
    v_over_cos_lat = grid.to_modal(v_nodal / grid.cos_lat)
    vorticity = grid.curl_cos_lat((u_over_cos_lat, v_over_cos_lat), clip=clip)
    divergence = grid.div_cos_lat((u_over_cos_lat, v_over_cos_lat), clip=clip)
    return vorticity, divergence


def _dfi_lanczos_weights(time_span, cutoff_period, dt):
    N = round(time_span / (2 * dt))
    n = np.arange(1, N + 1)
    w = np.sinc(n / (N + 1)) * np.sinc(n * time_span / (cutoff_period * N))
    return w


@dataclasses.dataclass
class TimeReversedImExODE(di.ImplicitExplicitODE):
    forward_eq: di.ImplicitExplicitODE

    def explicit_terms(self, state):
        forward_term = self.forward_eq.explicit_terms(state)
        return di.tree_map(jnp.negative, forward_term)

    def implicit_terms(self, state):
        forward_term = self.forward_eq.implicit_terms(state)
        return di.tree_map(jnp.negative, forward_term)

    def implicit_inverse(
        self,
        state,
        step_size: float,
    ):
        return self.forward_eq.implicit_inverse(state, -step_size)


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


def digital_filter_initialization(equation, ode_solver, filters, time_span,
                                  cutoff_period, dt):

    def f(state):
        forward_step = di.step_with_filters(ode_solver(equation, dt), filters)
        backward_step = di.step_with_filters(
            ode_solver(TimeReversedImExODE(equation), dt), filters)
        weights = _dfi_lanczos_weights(time_span, cutoff_period, dt)
        init_weight = 1.0
        total_weight = init_weight + 2 * weights.sum()
        init_weight /= total_weight
        weights /= total_weight
        init_term = di.tree_map(lambda x: x * init_weight, state)
        forward_term = accumulate_repeated(forward_step, weights, state)
        backward_term = accumulate_repeated(backward_step, weights, state)
        return di.tree_map(lambda *xs: sum(xs), init_term, forward_term,
                        backward_term)

    return f


layers = 32
ref_temp_si = 250 * units.degK
model_coords = di.CoordinateSystem(
    di.Grid(longitude_wavenumbers=171,
            total_wavenumbers=172,
            longitude_nodes=512,
            latitude_nodes=256),
    di.SigmaCoordinates(np.linspace(0, 1, layers + 1, dtype=np.float32)),
)
dt_si = 5 * units.minute
save_every = 15 * units.minute
total_time = 2 * units.day + save_every
dfi_timescale = 6 * units.hour
output_level_indices = [layers // 4, layers // 2, 3 * layers // 4, -1]

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
desired_lon = 180 / np.pi * model_coords.horizontal.nodal_axes[0]
desired_lat = 180 / np.pi * np.arcsin(model_coords.horizontal.nodal_axes[1])
ds_init = attach_xarray_units(ds.compute().interp(latitude=desired_lat,
                                                  longitude=desired_lon))
ds_init["orography"] = attach_data_array_units(
    raw_orography.interp(latitude=desired_lat, longitude=desired_lon))
ds_init["orography"] /= di.GRAVITY_ACCELERATION
source_vertical = HybridCoordinates()
ds_nondim_init = xarray_nondimensionalize(ds_init)
model_level_inputs = xarray_to_gcm_dict(ds_nondim_init)
sp_nodal = model_level_inputs.pop("surface_pressure")
orography_input = model_level_inputs.pop("orography")
sp_init_hpa = (ds_init.surface_pressure.transpose(
    "longitude", "latitude").data.to("hPa").magnitude)
nodal_inputs = regrid_hybrid_to_sigma(
    fields=model_level_inputs,
    hybrid_coords=source_vertical,
    sigma_coords=model_coords.vertical,
    surface_pressure=sp_init_hpa,
)
u_nodal = nodal_inputs["u_component_of_wind"]
v_nodal = nodal_inputs["v_component_of_wind"]
t_nodal = nodal_inputs["temperature"]
vorticity, divergence = uv_nodal_to_vor_div_modal(model_coords.horizontal,
                                                  u_nodal, v_nodal)
ref_temps = di.DEFAULT_SCALE.nondimensionalize(ref_temp_si * np.ones(
    (model_coords.vertical.layers, )))
assert ref_temps.shape == (model_coords.vertical.layers, )
temperature_variation = model_coords.horizontal.to_modal(
    t_nodal - ref_temps.reshape(-1, 1, 1))
log_sp = model_coords.horizontal.to_modal(np.log(sp_nodal))
tracers = model_coords.horizontal.to_modal({
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
orography = model_coords.horizontal.to_modal(orography_input)
orography = di.exponential_filter(model_coords.horizontal, order=2)(orography)
eq = di.PrimitiveEquations(ref_temps, orography, model_coords)
res_factor = model_coords.horizontal.latitude_nodes / 128
dt = di.DEFAULT_SCALE.nondimensionalize(dt_si)
tau = di.DEFAULT_SCALE.nondimensionalize(8.6 / (2.4**np.log2(res_factor)) *
                                         units.hours)
hyperdiffusion_filter = horizontal_diffusion_step_filter(
    model_coords.horizontal, dt=dt, tau=tau, order=2)
time_span = cutoff_period = di.DEFAULT_SCALE.nondimensionalize(dfi_timescale)
dfi = jax.jit(
    digital_filter_initialization(
        equation=eq,
        ode_solver=di.imex_rk_sil3,
        filters=[hyperdiffusion_filter],
        time_span=time_span,
        cutoff_period=cutoff_period,
        dt=dt,
    ))
dfi_init_state = jax.block_until_ready(dfi(raw_init_state))

inner_steps = int(save_every / dt_si)
outer_steps = int(total_time / save_every)
step_fn = di.step_with_filters(
    di.imex_rk_sil3(eq, dt),
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
ds_out
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
