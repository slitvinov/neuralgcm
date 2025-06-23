from typing import Any
import dataclasses
import functools
import jax
import jax.numpy as jnp
import numpy as np
import scipy
import tree_math

tree_map = jax.tree_util.tree_map
einsum = functools.partial(jnp.einsum, precision=jax.lax.Precision.HIGHEST)

# gravity_acceleration = DEFAULT_SCALE.nondimensionalize(GRAVITY_ACCELERATION)
gravity_acceleration = 7.2364082834567185e+01
kappa = 2 / 7
ideal_gas_constant = kappa * 0.0011628807950492582


class g:
    pass


def to_modal(z):
    return tree_map_over_nonscalars(transform, z)


def to_nodal(x):
    return tree_map_over_nonscalars(inverse_transform, x)


def transform(x):
    wx = g.w * x
    fwx = einsum("im,...ij->...mj", g.f, wx)
    pfwx = einsum("mjl,...mj->...ml", g.p, fwx)
    return pfwx


def inverse_transform(x):
    px = einsum("mjl,...ml->...mj", g.p, x)
    fpx = einsum("im,...mj->...ij", g.f, px)
    return fpx


def basis():
    dft = scipy.linalg.dft(
        g.longitude_nodes)[:, :g.longitude_wavenumbers] / np.sqrt(np.pi)
    f = np.empty((g.longitude_nodes, 2 * g.longitude_wavenumbers - 1))
    f[:, 0] = 1 / np.sqrt(2 * np.pi)
    f[:, 1::2] = np.real(dft[:, 1:])
    f[:, 2::2] = -np.imag(dft[:, 1:])
    x, w = scipy.special.roots_legendre(g.latitude_nodes)
    w *= 2 * np.pi / g.longitude_nodes
    q = np.sqrt(1 - x * x)
    y = np.zeros(
        (g.total_wavenumbers, g.longitude_wavenumbers, g.latitude_nodes))
    y[0, 0] = 1 / np.sqrt(2)
    for m in range(1, g.longitude_wavenumbers):
        y[0, m] = -np.sqrt(1 + 1 / (2 * m)) * q * y[0, m - 1]
    for k in range(1, g.total_wavenumbers):
        m_max = min(g.longitude_wavenumbers, g.total_wavenumbers - k)
        m = np.arange(m_max).reshape((-1, 1))
        m2 = np.square(m)
        mk2 = np.square(m + k)
        mkp2 = np.square(m + k - 1)
        a = np.sqrt((4 * mk2 - 1) / (mk2 - m2))
        b = np.sqrt((mkp2 - m2) / (4 * mkp2 - 1))
        y[k, :m_max] = a * (x * y[k - 1, :m_max] - b * y[k - 2, :m_max])
    r = np.transpose(y, (1, 2, 0))
    p = np.zeros(
        (g.longitude_wavenumbers, g.latitude_nodes, g.total_wavenumbers))
    for m in range(g.longitude_wavenumbers):
        p[m, :, m:g.total_wavenumbers] = r[m, :, 0:g.total_wavenumbers - m]
    p = np.repeat(p, 2, axis=0)
    return f, p[1:], w


def clip_wavenumbers(x):

    def clip(x):
        mask = jnp.ones(g.total_wavenumbers, x.dtype).at[-1:].set(0)
        return x * mask

    return tree_map_over_nonscalars(clip, x)


def pad_in_dim(x, pad_width, axis):
    padding_value = jnp.array(0, dtype=x.dtype)
    padding_config = [(0, 0, 0)] * x.ndim
    padding_config[axis] = pad_width + (0, )
    return jax.lax.pad(x, padding_value, padding_config)


def shift(x, offset, axis):
    if offset > 0:
        sliced = jax.lax.slice_in_dim(x, 0, x.shape[axis] - offset, axis=axis)
        return pad_in_dim(sliced, (offset, 0), axis=axis)
    else:
        sliced = jax.lax.slice_in_dim(x, -offset, x.shape[axis], axis=axis)
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
    dx = jax.lax.slice_in_dim(x, 1, None, axis=-3) - jax.lax.slice_in_dim(
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
    xd𝜎 = einsum(x, [0, 1, 2], g.layer_thickness, [0], [0, 1, 2])
    i = jnp.arange(di.g.layers)[:, jnp.newaxis]
    j = jnp.arange(di.g.layers)[jnp.newaxis, :]
    w = jnp.less_equal(i, j).astype(np.float32)
    return jnp.einsum(
        w,
        [0, 3],
        xd𝜎,
        [0, 1, 2],
        [3, 1, 2],
        precision=("bfloat16", "highest"),
    )


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
    return -0.5 * (jax.lax.slice_in_dim(w_times_x_diff, 1, None, axis=-3) +
                   jax.lax.slice_in_dim(w_times_x_diff, 0, -1, axis=-3))


def real_basis_derivative(u):
    i = jnp.arange(u.shape[-2]).reshape(-1, 1)
    j = (i + 1) // 2
    u_down = shift(u, -1, -2)
    u_up = shift(u, 1, -2)
    return j * jnp.where(i % 2, u_down, -u_up)


def nodal_axes():
    longitude = np.linspace(0, 2 * np.pi, g.longitude_nodes, endpoint=False)
    sin_latitude, _ = scipy.special.roots_legendre(g.latitude_nodes)
    return longitude, sin_latitude


def modal_axes():
    m_pos = np.arange(1, g.longitude_wavenumbers)
    m_pos_neg = np.stack([m_pos, -m_pos], axis=1).ravel()
    return np.concatenate([[0], m_pos_neg]), np.arange(g.total_wavenumbers)


def cos_lat():
    sin_lat, _ = scipy.special.roots_legendre(g.latitude_nodes)
    return np.sqrt(1 - sin_lat**2)


def sec2_lat():
    sin_lat, _ = scipy.special.roots_legendre(g.latitude_nodes)
    return 1 / (1 - sin_lat**2)


def laplacian_eigenvalues():
    l = np.arange(g.total_wavenumbers)
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
    m, l = np.meshgrid(*modal_axes(), indexing="ij")
    mask = abs(m) <= l
    a = np.sqrt(mask * (l**2 - m**2) / (4 * l**2 - 1))
    a[:, 0] = 0
    b = np.sqrt(mask * ((l + 1)**2 - m**2) / (4 * (l + 1)**2 - 1))
    b[:, -1] = 0
    return a, b


def cos_lat_d_dlat(x):
    _, l = np.meshgrid(*modal_axes(), indexing="ij")
    a, b = _derivative_recurrence_weights()
    x_lm1 = shift(((l + 1) * a) * x, -1, axis=-1)
    x_lp1 = shift((-l * b) * x, +1, axis=-1)
    return x_lm1 + x_lp1


def sec_lat_d_dlat_cos2(x):
    _, l = np.meshgrid(*modal_axes(), indexing="ij")
    a, b = _derivative_recurrence_weights()
    x_lm1 = shift(((l - 1) * a) * x, -1, axis=-1)
    x_lp1 = shift((-(l + 2) * b) * x, +1, axis=-1)
    return x_lm1 + x_lp1


def cos_lat_grad(x, clip=True):
    raw = real_basis_derivative(x), cos_lat_d_dlat(x)
    if clip:
        return clip_wavenumbers(raw)
    return raw


def k_cross(v):
    return -v[1], v[0]


def div_cos_lat(v, clip=True):
    raw = real_basis_derivative(v[0]) + sec_lat_d_dlat_cos2(v[1])
    if clip:
        return clip_wavenumbers(raw)
    return raw


def curl_cos_lat(v, clip=True):
    raw = real_basis_derivative(v[1]) - sec_lat_d_dlat_cos2(v[0])
    if clip:
        return clip_wavenumbers(raw)
    return raw


def get_cos_lat_vector(vorticity, divergence, clip=True):
    stream_function = inverse_laplacian(vorticity)
    velocity_potential = inverse_laplacian(divergence)
    return jax.tree_util.tree_map(
        lambda x, y: x + y,
        cos_lat_grad(velocity_potential, clip=clip),
        k_cross(cos_lat_grad(stream_function, clip=clip)),
    )


def imex_runge_kutta(exp, imp, inv, dt):
    F = tree_math.unwrap(exp)
    G = tree_math.unwrap(imp)
    G_inv = tree_math.unwrap(inv, vector_argnums=0)
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
        return y0 + ex_terms + im_terms

    return step_fn


@tree_math.struct
class State:
    vorticity: Any
    divergence: Any
    temperature_variation: Any
    log_surface_pressure: Any
    tracers: Any = dataclasses.field(default_factory=dict)


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
    sigma_dot_explicit = jax.lax.slice_in_dim(
        sum_𝜎 * jax.lax.slice_in_dim(f_explicit, -1, None) - f_explicit, 0, -1)
    sigma_dot_full = jax.lax.slice_in_dim(
        sum_𝜎 * jax.lax.slice_in_dim(f_full, -1, None) - f_full, 0, -1)
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


def get_temperature_implicit_weights():
    p = np.tril(np.ones([g.layers, g.layers]))
    alpha = get_sigma_ratios()[..., np.newaxis]
    p_alpha = p * alpha
    p_alpha_shifted = np.roll(p_alpha, 1, axis=0)
    p_alpha_shifted[0] = 0
    h0 = (kappa * g.reference_temperature[..., np.newaxis] *
          (p_alpha + p_alpha_shifted) / g.layer_thickness[..., np.newaxis])
    temp_diff = np.diff(g.reference_temperature)
    thickness_sum = g.layer_thickness[:-1] + g.layer_thickness[1:]
    k0 = np.concatenate((temp_diff / thickness_sum, [0]), axis=0)[...,
                                                                  np.newaxis]
    thickness_cumulative = np.cumsum(g.layer_thickness)[..., np.newaxis]
    k1 = p - thickness_cumulative
    k = k0 * k1
    k_shifted = np.roll(k, 1, axis=0)
    k_shifted[0] = 0
    return (h0 - k - k_shifted) * g.layer_thickness


def _vertical_matvec(a, x):
    return einsum("gh,...hml->...gml", a, x)


def matvec(a, x):
    return einsum("lgh,...hml->...gml", a, x)


def div_sec_lat(m_component, n_component):
    m_component = to_modal(m_component * sec2_lat())
    n_component = to_modal(n_component * sec2_lat())
    return div_cos_lat((m_component, n_component), clip=False)


def _t_omega_over_sigma_sp(temperature_field, g_term, v_dot_grad_log_sp):
    f = cumulative_sigma_integral(g_term)
    alpha = get_sigma_ratios()
    alpha = alpha[:, np.newaxis, np.newaxis]
    del_𝜎 = g.layer_thickness[:, np.newaxis, np.newaxis]
    padding = [(1, 0), (0, 0), (0, 0)]
    g_part = (alpha * f + jnp.pad(alpha * f, padding)[:-1, ...]) / del_𝜎
    return temperature_field * (v_dot_grad_log_sp - g_part)


def horizontal_scalar_advection(scalar, aux_state):
    u, v = aux_state.cos_lat_u
    nodal_terms = scalar * aux_state.divergence
    modal_terms = -div_sec_lat(u * scalar, v * scalar)
    return nodal_terms, modal_terms


def _make_filter_fn(scaling):
    rescale = lambda x: scaling * x if np.shape(x) == np.broadcast_shapes(
        np.shape(x), scaling.shape) else x
    return functools.partial(jax.tree_util.tree_map, rescale)


def explicit_terms(state):
    aux_state = compute_diagnostic_state(state)
    sec2_lat0 = sec2_lat()
    u, v = aux_state.cos_lat_u
    _, coriolis_parameter = np.meshgrid(*nodal_axes(), indexing="ij")
    total_vorticity = aux_state.vorticity + coriolis_parameter
    nodal_vorticity_u = -v * total_vorticity * sec2_lat0
    nodal_vorticity_v = u * total_vorticity * sec2_lat0
    d𝜎_dt = aux_state.sigma_dot_full
    sigma_dot_u = -centered_vertical_advection(d𝜎_dt, u)
    sigma_dot_v = -centered_vertical_advection(d𝜎_dt, v)
    rt = ideal_gas_constant * aux_state.temperature_variation
    grad_log_ps_u, grad_log_ps_v = aux_state.cos_lat_grad_log_sp
    vertical_term_u = (sigma_dot_u + rt * grad_log_ps_u) * sec2_lat0
    vertical_term_v = (sigma_dot_v + rt * grad_log_ps_v) * sec2_lat0
    combined_u = to_modal(nodal_vorticity_u + vertical_term_u)
    combined_v = to_modal(nodal_vorticity_v + vertical_term_v)
    vorticity_tendency = -curl_cos_lat((combined_u, combined_v), clip=False)
    divergence_dot = -div_cos_lat((combined_u, combined_v), clip=False)
    nodal_cos_lat_u2 = jnp.stack(aux_state.cos_lat_u)**2
    kinetic = nodal_cos_lat_u2.sum(0) * sec2_lat() / 2
    kinetic_energy_tendency = -laplacian(to_modal(kinetic))
    orography_tendency = -gravity_acceleration * laplacian(g.orography)
    horizontal_tendency_fn = functools.partial(horizontal_scalar_advection,
                                               aux_state=aux_state)
    dT_dt_horizontal_nodal, dT_dt_horizontal_modal = horizontal_tendency_fn(
        aux_state.temperature_variation)
    tracers_horizontal_nodal_and_modal = jax.tree_util.tree_map(
        horizontal_tendency_fn, aux_state.tracers)
    sigma_dot_explicit = aux_state.sigma_dot_explicit
    sigma_dot_full = aux_state.sigma_dot_full
    temperature_variation = aux_state.temperature_variation
    tendency = centered_vertical_advection(sigma_dot_full,
                                           temperature_variation)
    if np.unique(g.reference_temperature[..., np.newaxis,
                                         np.newaxis].ravel()).size > 1:
        tendency += centered_vertical_advection(
            sigma_dot_explicit, g.reference_temperature[..., np.newaxis,
                                                        np.newaxis])
    dT_dt_vertical = tendency
    g_explicit = aux_state.u_dot_grad_log_sp
    g_full = g_explicit + aux_state.divergence
    mean_t_part = _t_omega_over_sigma_sp(
        g.reference_temperature[..., np.newaxis, np.newaxis], g_explicit,
        aux_state.u_dot_grad_log_sp)
    variation_t_part = _t_omega_over_sigma_sp(aux_state.temperature_variation,
                                              g_full,
                                              aux_state.u_dot_grad_log_sp)
    dT_dt_adiabatic = kappa * (mean_t_part + variation_t_part)
    log_sp_tendency = -sigma_integral(aux_state.u_dot_grad_log_sp)
    sigma_dot_full = aux_state.sigma_dot_full
    vertical_tendency_fn = functools.partial(centered_vertical_advection,
                                             sigma_dot_full)
    tracers_vertical_nodal = jax.tree_util.tree_map(vertical_tendency_fn,
                                                    aux_state.tracers)
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
    )
    return clip_wavenumbers(tendency)


def implicit_terms(state):
    geopotential_diff = get_geopotential_diff(state.temperature_variation)
    rt_log_p = (ideal_gas_constant *
                g.reference_temperature[..., np.newaxis, np.newaxis] *
                state.log_surface_pressure)
    vorticity_implicit = jnp.zeros_like(state.vorticity)
    divergence_implicit = -laplacian(geopotential_diff + rt_log_p)
    weights = -get_temperature_implicit_weights()
    temperature_variation_implicit = _vertical_matvec(weights,
                                                      state.divergence)
    log_surface_pressure_implicit = -_vertical_matvec(
        g.layer_thickness[np.newaxis], state.divergence)
    tracers_implicit = jax.tree_util.tree_map(jnp.zeros_like, state.tracers)
    return State(
        vorticity=vorticity_implicit,
        divergence=divergence_implicit,
        temperature_variation=temperature_variation_implicit,
        log_surface_pressure=log_surface_pressure_implicit,
        tracers=tracers_implicit,
    )


def implicit_inverse(state, dt):
    eye = np.eye(g.layers)[np.newaxis]
    lam = laplacian_eigenvalues()
    geo = get_geopotential_weights()
    r = ideal_gas_constant
    h = get_temperature_implicit_weights()
    t = g.reference_temperature[:, np.newaxis]
    thickness = g.layer_thickness[np.newaxis, np.newaxis, :]
    l = g.total_wavenumbers
    j = k = g.layers
    row0 = np.concatenate(
        [
            np.broadcast_to(eye, [l, j, k]),
            dt * np.einsum("l,jk->ljk", lam, geo),
            dt * r * np.einsum("l,jo->ljo", lam, t),
        ],
        axis=2,
    )
    row1 = np.concatenate(
        [
            dt * np.broadcast_to(h[np.newaxis], [l, j, k]),
            np.broadcast_to(eye, [l, j, k]),
            np.zeros([l, j, 1]),
        ],
        axis=2,
    )
    row2 = np.concatenate(
        [
            dt * np.broadcast_to(thickness, [l, 1, k]),
            np.zeros([l, 1, k]),
            np.ones([l, 1, 1]),
        ],
        axis=2,
    )
    implicit_matrix = np.concatenate((row0, row1, row2), axis=1)
    inverse = np.linalg.inv(implicit_matrix)
    div = slice(0, g.layers)
    temp = slice(g.layers, 2 * g.layers)
    logp = slice(2 * g.layers, 2 * g.layers + 1)
    inverted_divergence = (
        matvec(inverse[:, div, div], state.divergence) +
        matvec(inverse[:, div, temp], state.temperature_variation) +
        matvec(inverse[:, div, logp], state.log_surface_pressure))
    inverted_temperature_variation = (
        matvec(inverse[:, temp, div], state.divergence) +
        matvec(inverse[:, temp, temp], state.temperature_variation) +
        matvec(inverse[:, temp, logp], state.log_surface_pressure))
    inverted_log_surface_pressure = (
        matvec(inverse[:, logp, div], state.divergence) +
        matvec(inverse[:, logp, temp], state.temperature_variation) +
        matvec(inverse[:, logp, logp], state.log_surface_pressure))
    return State(
        state.vorticity,
        inverted_divergence,
        inverted_temperature_variation,
        inverted_log_surface_pressure,
        state.tracers,
    )
