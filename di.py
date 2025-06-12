from __future__ import annotations
from jax import lax
from typing import Any
import dataclasses
import functools
import jax
import jax.numpy as jnp
import numpy as np
import scipy
import scipy.special as sps
import tree_math

tree_map = jax.tree_util.tree_map
einsum = functools.partial(jnp.einsum, precision=lax.Precision.HIGHEST)
_CONSTANT_NORMALIZATION_FACTOR = 3.5449077

# gravity_acceleration = DEFAULT_SCALE.nondimensionalize(GRAVITY_ACCELERATION)
gravity_acceleration = 7.2364082834567185e+01
kappa = 2 / 7
ideal_gas_constant = kappa * 0.0011628807950492582


class g:
    pass


@jax.jit
def uv_nodal_to_vor_div_modal(u_nodal, v_nodal):
    u_over_cos_lat = to_modal(u_nodal / cos_lat())
    v_over_cos_lat = to_modal(v_nodal / cos_lat())
    vorticity = curl_cos_lat((u_over_cos_lat, v_over_cos_lat), clip=True)
    divergence = div_cos_lat((u_over_cos_lat, v_over_cos_lat), clip=True)
    return vorticity, divergence


def to_modal(z):
    return tree_map_over_nonscalars(transform, z)


def to_nodal(x):
    return tree_map_over_nonscalars(inverse_transform, x)


def transform(x):
    f, p, w = basis()
    wx = w * x
    fwx = einsum("im,...ij->...mj", f, wx)
    pfwx = einsum("mjl,...mj->...ml", p, fwx)
    return pfwx


def inverse_transform(x):
    f, p, w = basis()
    px = einsum("mjl,...ml->...mj", p, x)
    fpx = einsum("im,...mj->...ij", f, px)
    return fpx


def basis():
    f = real_basis()
    wf = 2 * np.pi / g.longitude_nodes
    x, wp = scipy.special.roots_legendre(g.latitude_nodes)
    w = wf * wp
    p = evaluate(x)
    p = np.repeat(p, 2, axis=0)
    p = p[1:]
    return f, p, w


def clip_wavenumbers(x):

    def clip(x):
        modal_shape = 2 * g.longitude_wavenumbers - 1, g.total_wavenumbers
        mask = jnp.ones(modal_shape[-1], x.dtype).at[-1:].set(0)
        return x * mask

    return tree_map_over_nonscalars(clip, x)


def cumsum(x, axis):
    if axis < 0:
        axis = axis + x.ndim
    size = x.shape[axis]
    i = jnp.arange(size)[:, jnp.newaxis]
    j = jnp.arange(size)[jnp.newaxis, :]
    w = jnp.less_equal(i, j).astype(np.float32)
    out_axes = list(range(x.ndim))
    out_axes[axis] = x.ndim
    return jnp.einsum(
        w,
        [axis, x.ndim],
        x,
        list(range(x.ndim)),
        out_axes,
        precision=("bfloat16", "highest"),
    )


def pad_in_dim(x, pad_width, axis):
    padding_value = jnp.array(0, dtype=x.dtype)
    padding_config = [(0, 0, 0)] * x.ndim
    padding_config[axis] = pad_width + (0, )
    return lax.pad(x, padding_value, padding_config)


def shift(x, offset, axis):
    if offset > 0:
        sliced = lax.slice_in_dim(x, 0, x.shape[axis] - offset, axis=axis)
        return pad_in_dim(sliced, (offset, 0), axis=axis)
    else:
        sliced = lax.slice_in_dim(x, -offset, x.shape[axis], axis=axis)
        return pad_in_dim(sliced, (0, -offset), axis=axis)


def tree_map_over_nonscalars(f, x):

    def g(x):
        x = jnp.asarray(x)
        return f(x) if x.ndim else x

    return tree_map(g, x)


def _slice_shape_along_axis(x):
    x_shape = list(x.shape)
    x_shape[-3] = 1
    return tuple(x_shape)


def centered_difference(x):
    dx = lax.slice_in_dim(x, 1, None, axis=-3) - lax.slice_in_dim(
        x, 0, -1, axis=-3)
    dx_axes = range(dx.ndim)
    inv_d𝜎 = 1 / g.center_to_center
    inv_d𝜎_axes = [dx_axes[-3]]
    return einsum(dx,
                  dx_axes,
                  inv_d𝜎,
                  inv_d𝜎_axes,
                  dx_axes,
                  precision="float32")


def cumulative_sigma_integral(x):
    x_axes = range(x.ndim)
    d𝜎 = g.layer_thickness
    d𝜎_axes = [x_axes[-3]]
    xd𝜎 = einsum(x, x_axes, d𝜎, d𝜎_axes, x_axes)
    return cumsum(xd𝜎, -3)


def sigma_integral(x):
    x_axes = range(x.ndim)
    d𝜎 = g.layer_thickness
    d𝜎_axes = [x_axes[-3]]
    xd𝜎 = einsum(x, x_axes, d𝜎, d𝜎_axes, x_axes)
    return xd𝜎.sum(axis=-3, keepdims=True)


def centered_vertical_advection(w, x):
    w_slc_shape = _slice_shape_along_axis(w)
    w_boundary_values = (
        jnp.zeros(w_slc_shape, dtype=jax.dtypes.canonicalize_dtype(w.dtype)),
        jnp.zeros(w_slc_shape, dtype=jax.dtypes.canonicalize_dtype(w.dtype)),
    )
    x_slc_shape = _slice_shape_along_axis(x)
    dx_dsigma_boundary_values = (
        jnp.zeros(x_slc_shape, dtype=jax.dtypes.canonicalize_dtype(x.dtype)),
        jnp.zeros(x_slc_shape, dtype=jax.dtypes.canonicalize_dtype(x.dtype)),
    )
    w_boundary_top, w_boundary_bot = w_boundary_values
    w = jnp.concatenate([w_boundary_top, w, w_boundary_bot], axis=-3)
    x_diff = centered_difference(x)
    x_diff_boundary_top, x_diff_boundary_bot = dx_dsigma_boundary_values
    x_diff = jnp.concatenate(
        [x_diff_boundary_top, x_diff, x_diff_boundary_bot], axis=-3)
    w_times_x_diff = w * x_diff
    return -0.5 * (lax.slice_in_dim(w_times_x_diff, 1, None, axis=-3) +
                   lax.slice_in_dim(w_times_x_diff, 0, -1, axis=-3))


def evaluate_rhombus(x):
    n_m = g.longitude_wavenumbers
    n_l = g.total_wavenumbers
    y = np.sqrt(1 - x * x)
    p = np.zeros((n_l, n_m, len(x)))
    p[0, 0] = p[0, 0] + 1 / np.sqrt(2)
    for m in range(1, n_m):
        p[0, m] = -np.sqrt(1 + 1 / (2 * m)) * y * p[0, m - 1]
    m_max = n_m
    for k in range(1, n_l):
        m_max = min(n_m, n_l - k)
        m = np.arange(m_max).reshape((-1, 1))
        m2 = np.square(m)
        mk2 = np.square(m + k)
        mkp2 = np.square(m + k - 1)
        a = np.sqrt((4 * mk2 - 1) / (mk2 - m2))
        b = np.sqrt((mkp2 - m2) / (4 * mkp2 - 1))
        p[k, :m_max] = a * (x * p[k - 1, :m_max] - b * p[k - 2, :m_max])
    return p


def evaluate(x):
    n_m = g.longitude_wavenumbers
    n_l = g.total_wavenumbers
    r = np.transpose(evaluate_rhombus(x), (1, 2, 0))
    p = np.zeros((n_m, len(x), n_l))
    for m in range(n_m):
        p[m, :, m:n_l] = r[m, :, 0:n_l - m]
    return p


def real_basis():
    wavenumbers = g.longitude_wavenumbers
    nodes = g.longitude_nodes
    dft = scipy.linalg.dft(nodes)[:, :wavenumbers] / np.sqrt(np.pi)
    cos = np.real(dft[:, 1:])
    sin = -np.imag(dft[:, 1:])
    f = np.empty(shape=[nodes, 2 * wavenumbers - 1], dtype=np.float64)
    f[:, 0] = 1 / np.sqrt(2 * np.pi)
    f[:, 1::2] = cos
    f[:, 2::2] = sin
    return f


def real_basis_derivative(u):
    axis = -2
    i = jnp.arange(u.shape[axis]).reshape((-1, ) + (1, ) * (-1 - axis))
    j = (i + 1) // 2
    u_down = shift(u, -1, axis)
    u_up = shift(u, +1, axis)
    return j * jnp.where(i % 2, u_down, -u_up)


def nodal_axes():
    longitude = np.linspace(0, 2 * np.pi, g.longitude_nodes, endpoint=False)
    sin_latitude, _ = sps.roots_legendre(g.latitude_nodes)
    return longitude, sin_latitude


def nodal_mesh():
    return np.meshgrid(*nodal_axes(), indexing="ij")


def modal_axes():
    m_pos = np.arange(1, g.longitude_wavenumbers)
    m_pos_neg = np.stack([m_pos, -m_pos], axis=1).ravel()
    lon_wavenumbers = np.concatenate([[0], m_pos_neg])
    tot_wavenumbers = np.arange(g.total_wavenumbers)
    return lon_wavenumbers, tot_wavenumbers


def modal_shape():
    return 2 * g.longitude_wavenumbers - 1, g.total_wavenumbers


def mask():
    m, l = np.meshgrid(*modal_axes(), indexing="ij")
    return abs(m) <= l


def modal_mesh():
    return np.meshgrid(*modal_axes(), indexing="ij")


def cos_lat():
    _, sin_lat = nodal_axes()
    return np.sqrt(1 - sin_lat**2)


def sec2_lat():
    _, sin_lat = nodal_axes()
    return 1 / (1 - sin_lat**2)


def laplacian_eigenvalues():
    _, l = modal_axes()
    return -l * (l + 1) / (1.0**2)


def laplacian(x):
    return x * laplacian_eigenvalues()


def inverse_laplacian(x):
    with np.errstate(divide="ignore", invalid="ignore"):
        inverse_eigenvalues = 1 / laplacian_eigenvalues()
    inverse_eigenvalues[0] = 0
    inverse_eigenvalues[g.total_wavenumbers:] = 0
    assert not np.isnan(inverse_eigenvalues).any()
    return x * inverse_eigenvalues


def _derivative_recurrence_weights():
    m, l = modal_mesh()
    a = np.sqrt(mask() * (l**2 - m**2) / (4 * l**2 - 1))
    a[:, 0] = 0
    b = np.sqrt(mask() * ((l + 1)**2 - m**2) / (4 * (l + 1)**2 - 1))
    b[:, -1] = 0
    return a, b


def cos_lat_d_dlat(x):
    _, l = modal_mesh()
    a, b = _derivative_recurrence_weights()
    x_lm1 = shift(((l + 1) * a) * x, -1, axis=-1)
    x_lp1 = shift((-l * b) * x, +1, axis=-1)
    return x_lm1 + x_lp1


def sec_lat_d_dlat_cos2(x):
    _, l = modal_mesh()
    a, b = _derivative_recurrence_weights()
    x_lm1 = shift(((l - 1) * a) * x, -1, axis=-1)
    x_lp1 = shift((-(l + 2) * b) * x, +1, axis=-1)
    return x_lm1 + x_lp1


def cos_lat_grad(x, clip=True):
    raw = real_basis_derivative(x) / 1.0, cos_lat_d_dlat(x) / 1.0
    if clip:
        return clip_wavenumbers(raw)
    return raw


def k_cross(v):
    return -v[1], v[0]


def div_cos_lat(v, clip=True):
    raw = (real_basis_derivative(v[0]) + sec_lat_d_dlat_cos2(v[1])) / 1.0
    if clip:
        return clip_wavenumbers(raw)
    return raw


def curl_cos_lat(v, clip=True):
    raw = (real_basis_derivative(v[1]) - sec_lat_d_dlat_cos2(v[0])) / 1.0
    if clip:
        return clip_wavenumbers(raw)
    return raw


def add_constant(x: jnp.ndarray, c):
    return x.at[..., 0, 0].add(_CONSTANT_NORMALIZATION_FACTOR * c)


def get_cos_lat_vector(vorticity, divergence, clip=True):
    stream_function = inverse_laplacian(vorticity)
    velocity_potential = inverse_laplacian(divergence)
    return jax.tree_util.tree_map(
        lambda x, y: x + y,
        cos_lat_grad(velocity_potential, clip=clip),
        k_cross(cos_lat_grad(stream_function, clip=clip)),
    )


class ImplicitExplicitODE:

    def __init__(self, explicit_terms, implicit_terms, implicit_inverse):
        self.explicit_terms = explicit_terms
        self.implicit_terms = implicit_terms
        self.implicit_inverse = implicit_inverse


def imex_runge_kutta(equation, time_step):
    dt = time_step
    F = tree_math.unwrap(equation.explicit_terms)
    G = tree_math.unwrap(equation.implicit_terms)
    G_inv = tree_math.unwrap(equation.implicit_inverse, vector_argnums=0)
    a_ex = [[1 / 3], [1 / 6, 1 / 2], [1 / 2, -1 / 2, 1]]
    a_im = [[1 / 6, 1 / 6], [1 / 3, 0, 1 / 3], [3 / 8, 0, 3 / 8, 1 / 4]]
    b_ex = [1 / 2, -1 / 2, 1, 0]
    b_im = [3 / 8, 0, 3 / 8, 1 / 4]
    num_steps = len(b_ex)

    @tree_math.wrap
    def step_fn(y0):
        f = [None] * num_steps
        g = [None] * num_steps
        f[0] = F(y0)
        g[0] = G(y0)
        for i in range(1, num_steps):
            ex_terms = dt * sum(a_ex[i - 1][j] * f[j]
                                for j in range(i) if a_ex[i - 1][j])
            im_terms = dt * sum(a_im[i - 1][j] * g[j]
                                for j in range(i) if a_im[i - 1][j])
            Y_star = y0 + ex_terms + im_terms
            Y = G_inv(Y_star, dt * a_im[i - 1][i])
            if any(a_ex[j][i] for j in range(i, num_steps - 1)) or b_ex[i]:
                f[i] = F(Y)
            if any(a_im[j][i] for j in range(i, num_steps - 1)) or b_im[i]:
                g[i] = G(Y)
        ex_terms = dt * sum(b_ex[j] * f[j]
                            for j in range(num_steps) if b_ex[j])
        im_terms = dt * sum(b_im[j] * g[j]
                            for j in range(num_steps) if b_im[j])
        y_next = y0 + ex_terms + im_terms
        return y_next

    return step_fn


def runge_kutta_step_filter(state_filter):

    def _filter(u, u_next):
        del u
        return state_filter(u_next)

    return _filter


def exponential_filter(total_wavenumbers, attenuation=16, order=18, cutoff=0):
    total_wavenumber = np.arange(total_wavenumbers)
    k = total_wavenumber / total_wavenumber.max()
    a = attenuation
    c = cutoff
    p = order
    scaling = jnp.exp((k > c) * (-a * (((k - c) / (1 - c))**(2 * p))))
    return _make_filter_fn(scaling)


def exponential_step_filter(total_wavenumbers,
                            dt,
                            tau=0.010938,
                            order=18,
                            cutoff=0):
    filter_fn = exponential_filter(total_wavenumbers, dt / tau, order, cutoff)
    return runge_kutta_step_filter(filter_fn)


def step_with_filters(step_fn, filters):

    def _step_fn(u):
        u_next = step_fn(u)
        for filter_fn in filters:
            u_next = filter_fn(u, u_next)
        return u_next

    return _step_fn


def repeated(fn, steps, scan_fn=jax.lax.scan):

    def f_repeated(x_initial):
        gfun = lambda x, _: (fn(x), None)
        x_final, _ = scan_fn(gfun, x_initial, xs=None, length=steps)
        return x_final

    return f_repeated


def trajectory_from_step(
    step_fn,
    outer_steps,
    inner_steps,
    *,
    start_with_input=False,
    post_process_fn=lambda x: x,
):
    if inner_steps != 1:
        step_fn = repeated(step_fn, inner_steps, jax.lax.scan)

    def step(carry_in, _):
        carry_out = step_fn(carry_in)
        frame = carry_in if start_with_input else carry_out
        return carry_out, post_process_fn(frame)

    def multistep(x):
        return jax.lax.scan(step, x, xs=None, length=outer_steps)

    return multistep


@tree_math.struct
class State:
    vorticity: Any
    divergence: Any
    temperature_variation: Any
    log_surface_pressure: Any
    tracers: Any = dataclasses.field(default_factory=dict)
    sim_time: Any = None


@tree_math.struct
class DiagnosticState:
    vorticity: Any
    divergence: Any
    temperature_variation: Any
    cos_lat_u: Any
    sigma_dot_explicit: Any
    sigma_dot_full: Any
    cos_lat_grad_log_sp: Any
    u_dot_grad_log_sp: Any
    tracers: Any


def compute_diagnostic_state(state):

    nodal_vorticity = to_nodal(state.vorticity)
    nodal_divergence = to_nodal(state.divergence)
    nodal_temperature_variation = to_nodal(state.temperature_variation)
    tracers = to_nodal(state.tracers)
    nodal_cos_lat_u = jax.tree_util.tree_map(
        to_nodal,
        get_cos_lat_vector(state.vorticity, state.divergence, clip=False),
    )
    cos_lat_grad_log_sp = cos_lat_grad(state.log_surface_pressure, clip=False)
    nodal_cos_lat_grad_log_sp = to_nodal(cos_lat_grad_log_sp)
    nodal_u_dot_grad_log_sp = sum(
        jax.tree_util.tree_map(
            lambda x, y: x * y * sec2_lat(),
            nodal_cos_lat_u,
            nodal_cos_lat_grad_log_sp,
        ))
    f_explicit = cumulative_sigma_integral(nodal_u_dot_grad_log_sp)
    f_full = cumulative_sigma_integral(nodal_divergence +
                                       nodal_u_dot_grad_log_sp)
    sum_𝜎 = np.cumsum(g.layer_thickness)[:, np.newaxis, np.newaxis]
    sigma_dot_explicit = lax.slice_in_dim(
        sum_𝜎 * lax.slice_in_dim(f_explicit, -1, None) - f_explicit, 0, -1)
    sigma_dot_full = lax.slice_in_dim(
        sum_𝜎 * lax.slice_in_dim(f_full, -1, None) - f_full, 0, -1)
    return DiagnosticState(
        vorticity=nodal_vorticity,
        divergence=nodal_divergence,
        temperature_variation=nodal_temperature_variation,
        cos_lat_u=nodal_cos_lat_u,
        sigma_dot_explicit=sigma_dot_explicit,
        sigma_dot_full=sigma_dot_full,
        cos_lat_grad_log_sp=nodal_cos_lat_grad_log_sp,
        u_dot_grad_log_sp=nodal_u_dot_grad_log_sp,
        tracers=tracers,
    )


def get_sigma_ratios():
    alpha = np.diff(np.log(g.centers), append=0) / 2
    alpha[-1] = -np.log(g.centers[-1])
    return alpha


def get_geopotential_weights():
    alpha = get_sigma_ratios()
    weights = np.zeros([g.layers, g.layers])
    for j in range(g.layers):
        weights[j, j] = alpha[j]
        for k in range(j + 1, g.layers):
            weights[j, k] = alpha[k] + alpha[k - 1]
    return ideal_gas_constant * weights


def get_geopotential_diff(temperature):
    weights = get_geopotential_weights()
    return _vertical_matvec(weights, temperature)


def get_geopotential(temperature_variation, reference_temperature, orography):
    surface_geopotential = orography * gravity_acceleration
    temperature = add_constant(temperature_variation, reference_temperature)
    geopotential_diff = get_geopotential_diff(temperature)
    return surface_geopotential + geopotential_diff


def get_temperature_implicit_weights(reference_temperature):
    p = np.tril(np.ones([g.layers, g.layers]))
    alpha = get_sigma_ratios()[..., np.newaxis]
    p_alpha = p * alpha
    p_alpha_shifted = np.roll(p_alpha, 1, axis=0)
    p_alpha_shifted[0] = 0
    h0 = (kappa * reference_temperature[..., np.newaxis] *
          (p_alpha + p_alpha_shifted) / g.layer_thickness[..., np.newaxis])
    temp_diff = np.diff(reference_temperature)
    thickness_sum = g.layer_thickness[:-1] + g.layer_thickness[1:]
    k0 = np.concatenate((temp_diff / thickness_sum, [0]), axis=0)[...,
                                                                  np.newaxis]
    thickness_cumulative = np.cumsum(g.layer_thickness)[..., np.newaxis]
    k1 = p - thickness_cumulative
    k = k0 * k1
    k_shifted = np.roll(k, 1, axis=0)
    k_shifted[0] = 0
    return (h0 - k - k_shifted) * g.layer_thickness


def get_temperature_implicit(divergence, reference_temperature):
    weights = -get_temperature_implicit_weights(reference_temperature)
    return _vertical_matvec(weights, divergence)


def _vertical_matvec(a, x):
    return einsum("gh,...hml->...gml", a, x)


def _vertical_matvec_per_wavenumber(a, x):
    return einsum("lgh,...hml->...gml", a, x)


def div_sec_lat(m_component, n_component):
    m_component = to_modal(m_component * sec2_lat())
    n_component = to_modal(n_component * sec2_lat())
    return div_cos_lat((m_component, n_component), clip=False)


class PrimitiveEquations:

    def __init__(self, reference_temperature, orography):
        self.reference_temperature = reference_temperature
        self.orography = orography

    @property
    def coriolis_parameter(self):
        _, sin_lat = nodal_mesh()
        return sin_lat

    @property
    def T_ref(self):
        return self.reference_temperature[..., np.newaxis, np.newaxis]

    def _vertical_tendency(self, w, x):
        return centered_vertical_advection(w, x)

    def _t_omega_over_sigma_sp(self, temperature_field, g_term,
                               v_dot_grad_log_sp):
        f = cumulative_sigma_integral(g_term)
        alpha = get_sigma_ratios()
        alpha = alpha[:, np.newaxis, np.newaxis]
        del_𝜎 = g.layer_thickness[:, np.newaxis, np.newaxis]
        padding = [(1, 0), (0, 0), (0, 0)]
        g_part = (alpha * f + jnp.pad(alpha * f, padding)[:-1, ...]) / del_𝜎
        return temperature_field * (v_dot_grad_log_sp - g_part)

    def kinetic_energy_tendency(self, aux_state):
        nodal_cos_lat_u2 = jnp.stack(aux_state.cos_lat_u)**2
        kinetic = nodal_cos_lat_u2.sum(0) * sec2_lat() / 2
        return -laplacian(to_modal(kinetic))

    def orography_tendency(self):
        return -gravity_acceleration * laplacian(self.orography)

    def nodal_temperature_vertical_tendency(self, aux_state):
        sigma_dot_explicit = aux_state.sigma_dot_explicit
        sigma_dot_full = aux_state.sigma_dot_full
        temperature_variation = aux_state.temperature_variation
        tendency = self._vertical_tendency(sigma_dot_full,
                                           temperature_variation)
        if np.unique(self.T_ref.ravel()).size > 1:
            tendency += self._vertical_tendency(sigma_dot_explicit, self.T_ref)
        return tendency

    def horizontal_scalar_advection(self, scalar, aux_state):
        u, v = aux_state.cos_lat_u
        nodal_terms = scalar * aux_state.divergence
        modal_terms = -div_sec_lat(u * scalar, v * scalar)
        return nodal_terms, modal_terms

    def nodal_temperature_adiabatic_tendency(self, aux_state):
        g_explicit = aux_state.u_dot_grad_log_sp
        g_full = g_explicit + aux_state.divergence
        mean_t_part = self._t_omega_over_sigma_sp(self.T_ref, g_explicit,
                                                  aux_state.u_dot_grad_log_sp)
        variation_t_part = self._t_omega_over_sigma_sp(
            aux_state.temperature_variation, g_full,
            aux_state.u_dot_grad_log_sp)
        return kappa * (mean_t_part + variation_t_part)

    def explicit_terms(self, state):
        aux_state = compute_diagnostic_state(state)
        sec2_lat0 = sec2_lat()
        u, v = aux_state.cos_lat_u
        total_vorticity = aux_state.vorticity + self.coriolis_parameter
        nodal_vorticity_u = -v * total_vorticity * sec2_lat0
        nodal_vorticity_v = u * total_vorticity * sec2_lat0
        d𝜎_dt = aux_state.sigma_dot_full
        sigma_dot_u = -self._vertical_tendency(d𝜎_dt, u)
        sigma_dot_v = -self._vertical_tendency(d𝜎_dt, v)
        rt = ideal_gas_constant * aux_state.temperature_variation
        grad_log_ps_u, grad_log_ps_v = aux_state.cos_lat_grad_log_sp
        vertical_term_u = (sigma_dot_u + rt * grad_log_ps_u) * sec2_lat0
        vertical_term_v = (sigma_dot_v + rt * grad_log_ps_v) * sec2_lat0
        combined_u = to_modal(nodal_vorticity_u + vertical_term_u)
        combined_v = to_modal(nodal_vorticity_v + vertical_term_v)
        vorticity_tendency = -curl_cos_lat(
            (combined_u, combined_v), clip=False)
        divergence_dot = -div_cos_lat((combined_u, combined_v), clip=False)

        kinetic_energy_tendency = self.kinetic_energy_tendency(aux_state)
        orography_tendency = self.orography_tendency()
        horizontal_tendency_fn = functools.partial(
            self.horizontal_scalar_advection, aux_state=aux_state)
        dT_dt_horizontal_nodal, dT_dt_horizontal_modal = horizontal_tendency_fn(
            aux_state.temperature_variation)
        tracers_horizontal_nodal_and_modal = jax.tree_util.tree_map(
            horizontal_tendency_fn, aux_state.tracers)
        dT_dt_vertical = self.nodal_temperature_vertical_tendency(aux_state)
        dT_dt_adiabatic = self.nodal_temperature_adiabatic_tendency(aux_state)
        log_sp_tendency = -sigma_integral(aux_state.u_dot_grad_log_sp)
        sigma_dot_full = aux_state.sigma_dot_full
        vertical_tendency_fn = functools.partial(self._vertical_tendency,
                                                 sigma_dot_full)
        tracers_vertical_nodal = jax.tree_util.tree_map(
            vertical_tendency_fn, aux_state.tracers)
        to_modal_fn = to_modal
        divergence_tendency = (divergence_dot + kinetic_energy_tendency +
                               orography_tendency)
        temperature_tendency = (to_modal_fn(dT_dt_horizontal_nodal +
                                            dT_dt_vertical + dT_dt_adiabatic) +
                                dT_dt_horizontal_modal)
        log_surface_pressure_tendency = to_modal_fn(log_sp_tendency)
        tracers_tendency = jax.tree_util.tree_map(
            lambda x, y_z: to_modal_fn(x + y_z[0]) + y_z[1],
            tracers_vertical_nodal,
            tracers_horizontal_nodal_and_modal,
        )
        tendency = State(
            vorticity=vorticity_tendency,
            divergence=divergence_tendency,
            temperature_variation=temperature_tendency,
            log_surface_pressure=log_surface_pressure_tendency,
            tracers=tracers_tendency,
            sim_time=None if state.sim_time is None else 1.0,
        )
        return clip_wavenumbers(tendency)

    def implicit_terms(self, state):
        geopotential_diff = get_geopotential_diff(state.temperature_variation)
        rt_log_p = (ideal_gas_constant * self.T_ref *
                    state.log_surface_pressure)
        vorticity_implicit = jnp.zeros_like(state.vorticity)
        divergence_implicit = -laplacian(geopotential_diff + rt_log_p)
        temperature_variation_implicit = get_temperature_implicit(
            state.divergence,
            self.reference_temperature,
        )
        log_surface_pressure_implicit = -_vertical_matvec(
            g.layer_thickness[np.newaxis], state.divergence)
        tracers_implicit = jax.tree_util.tree_map(jnp.zeros_like,
                                                  state.tracers)
        return State(
            vorticity=vorticity_implicit,
            divergence=divergence_implicit,
            temperature_variation=temperature_variation_implicit,
            log_surface_pressure=log_surface_pressure_implicit,
            tracers=tracers_implicit,
            sim_time=None if state.sim_time is None else 0.0,
        )

    def implicit_inverse(self, state, step_size):
        eye = np.eye(g.layers)[np.newaxis]
        lam = laplacian_eigenvalues()
        geo = get_geopotential_weights()
        r = ideal_gas_constant
        h = get_temperature_implicit_weights(self.reference_temperature)
        t = self.reference_temperature[:, np.newaxis]
        thickness = g.layer_thickness[np.newaxis, np.newaxis, :]
        l = modal_shape()[1]
        j = k = g.layers
        row0 = np.concatenate(
            [
                np.broadcast_to(eye, [l, j, k]),
                step_size * np.einsum("l,jk->ljk", lam, geo),
                step_size * r * np.einsum("l,jo->ljo", lam, t),
            ],
            axis=2,
        )
        row1 = np.concatenate(
            [
                step_size * np.broadcast_to(h[np.newaxis], [l, j, k]),
                np.broadcast_to(eye, [l, j, k]),
                np.zeros([l, j, 1]),
            ],
            axis=2,
        )
        row2 = np.concatenate(
            [
                np.broadcast_to(step_size * thickness, [l, 1, k]),
                np.zeros([l, 1, k]),
                np.ones([l, 1, 1]),
            ],
            axis=2,
        )
        implicit_matrix = np.concatenate((row0, row1, row2), axis=1)
        assert implicit_matrix.dtype == np.float64
        layers = g.layers
        div = slice(0, layers)
        temp = slice(layers, 2 * layers)
        logp = slice(2 * layers, 2 * layers + 1)

        inverse = np.linalg.inv(implicit_matrix)
        assert not np.isnan(inverse).any()
        inverted_divergence = (
            _vertical_matvec_per_wavenumber(inverse[:, div, div],
                                            state.divergence) +
            _vertical_matvec_per_wavenumber(inverse[:, div, temp],
                                            state.temperature_variation) +
            _vertical_matvec_per_wavenumber(inverse[:, div, logp],
                                            state.log_surface_pressure))
        inverted_temperature_variation = (
            _vertical_matvec_per_wavenumber(inverse[:, temp, div],
                                            state.divergence) +
            _vertical_matvec_per_wavenumber(inverse[:, temp, temp],
                                            state.temperature_variation) +
            _vertical_matvec_per_wavenumber(inverse[:, temp, logp],
                                            state.log_surface_pressure))
        inverted_log_surface_pressure = (
            _vertical_matvec_per_wavenumber(inverse[:, logp, div],
                                            state.divergence) +
            _vertical_matvec_per_wavenumber(inverse[:, logp, temp],
                                            state.temperature_variation) +
            _vertical_matvec_per_wavenumber(inverse[:, logp, logp],
                                            state.log_surface_pressure))
        return State(
            state.vorticity,
            inverted_divergence,
            inverted_temperature_variation,
            inverted_log_surface_pressure,
            state.tracers,
            sim_time=state.sim_time,
        )


def _make_filter_fn(scaling):
    rescale = lambda x: scaling * x if np.shape(x) == np.broadcast_shapes(
        np.shape(x), scaling.shape) else x
    return functools.partial(jax.tree_util.tree_map, rescale)
