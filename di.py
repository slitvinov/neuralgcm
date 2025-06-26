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


def cumulative_sigma_integral(x):
    return jnp.cumsum(x * g.layer_thickness[:, None, None], axis=0)

def sigma_integral(x):
    x_axes = range(x.ndim)
    d𝜎 = g.layer_thickness
    d𝜎_axes = [x_axes[-3]]
    xd𝜎 = einsum(x, x_axes, d𝜎, d𝜎_axes, x_axes)
    return xd𝜎.sum(axis=-3, keepdims=True)


def centered_vertical_advection(w, x):
    w_slc_shape = _slice_shape_along_axis(w)
    x_slc_shape = _slice_shape_along_axis(x)
    w_boundary_top = jnp.zeros(w_slc_shape)
    w_boundary_bot = jnp.zeros(w_slc_shape)
    w = jnp.concatenate([w_boundary_top, w, w_boundary_bot], axis=-3)
    dx = jax.lax.slice_in_dim(x, 1, None, axis=-3) - jax.lax.slice_in_dim(
        x, 0, -1, axis=-3)
    inv_d𝜎 = 1 / g.center_to_center
    x_diff = einsum(dx, [0, 1, 2], inv_d𝜎, [0], [0, 1, 2], precision="float32")
    x_diff_boundary_top = jnp.zeros(x_slc_shape)
    x_diff_boundary_bot = jnp.zeros(x_slc_shape)
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


def laplacian_eigenvalues():
    l = np.arange(g.total_wavenumbers)
    return -l * (l + 1)


def laplacian(x):
    return x * laplacian_eigenvalues()


def derivative_recurrence_weights():
    m, l = np.meshgrid(*modal_axes(), indexing="ij")
    mask = abs(m) <= l
    a = np.sqrt(mask * (l**2 - m**2) / (4 * l**2 - 1))
    a[:, 0] = 0
    b = np.sqrt(mask * ((l + 1)**2 - m**2) / (4 * (l + 1)**2 - 1))
    b[:, -1] = 0
    return a, b


def cos_lat_d_dlat(x):
    _, l = np.meshgrid(*modal_axes(), indexing="ij")
    a, b = derivative_recurrence_weights()
    x_lm1 = shift(((l + 1) * a) * x, -1, axis=-1)
    x_lp1 = shift((-l * b) * x, +1, axis=-1)
    return x_lm1 + x_lp1


def sec_lat_d_dlat_cos2(x):
    _, l = np.meshgrid(*modal_axes(), indexing="ij")
    a, b = derivative_recurrence_weights()
    x_lm1 = shift(((l - 1) * a) * x, -1, axis=-1)
    x_lp1 = shift((-(l + 2) * b) * x, +1, axis=-1)
    return x_lm1 + x_lp1


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


def imex_runge_kutta(exp, imp, inv, dt):
    F = tree_math.unwrap(exp)
    G = tree_math.unwrap(imp)
    G_inv = tree_math.unwrap(inv, vector_argnums=0)
    a_ex = [[1 / 3], [1 / 6, 1 / 2], [1 / 2, -1 / 2, 1]]
    a_im = [[1 / 6, 1 / 6], [1 / 3, 0, 1 / 3], [3 / 8, 0, 3 / 8, 1 / 4]]
    b_ex = [1 / 2, -1 / 2, 1, 0]
    b_im = [3 / 8, 0, 3 / 8, 1 / 4]
    n = len(b_ex)

    @tree_math.wrap
    def step_fn(y0):
        f = [None] * n
        g = [None] * n
        f[0] = F(y0)
        g[0] = G(y0)
        for i in range(1, n):
            ex = dt * sum(a_ex[i - 1][j] * f[j]
                          for j in range(i) if a_ex[i - 1][j])
            im = dt * sum(a_im[i - 1][j] * g[j]
                          for j in range(i) if a_im[i - 1][j])
            Y = G_inv(y0 + ex + im, dt * a_im[i - 1][i])
            if any(a_ex[j][i] for j in range(i, n - 1)) or b_ex[i]: f[i] = F(Y)
            if any(a_im[j][i] for j in range(i, n - 1)) or b_im[i]: g[i] = G(Y)
        ex = dt * sum(b_ex[j] * f[j] for j in range(n) if b_ex[j])
        im = dt * sum(b_im[j] * g[j] for j in range(n) if b_im[j])
        return y0 + ex + im

    return step_fn


@tree_math.struct
class State:
    vorticity: Any
    divergence: Any
    temperature_variation: Any
    log_surface_pressure: Any
    tracers: Any = dataclasses.field(default_factory=dict)


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


def _t_omega_over_sigma_sp(temperature_field, g_term, v_dot_grad_log_sp):
    f = cumulative_sigma_integral(g_term)
    alpha = get_sigma_ratios()
    alpha = alpha[:, np.newaxis, np.newaxis]
    del_𝜎 = g.layer_thickness[:, np.newaxis, np.newaxis]
    padding = [(1, 0), (0, 0), (0, 0)]
    g_part = (alpha * f + jnp.pad(alpha * f, padding)[:-1, ...]) / del_𝜎
    return temperature_field * (v_dot_grad_log_sp - g_part)


def horizontal_scalar_advection(scalar, cos_lat_u, divergence):
    u, v = cos_lat_u
    nodal_terms = scalar * divergence
    sin_lat, _ = scipy.special.roots_legendre(g.latitude_nodes)
    sec2 = 1 / (1 - sin_lat**2)
    m_component = to_modal(u * scalar * sec2)
    n_component = to_modal(v * scalar * sec2)
    modal_terms = -div_cos_lat((m_component, n_component), clip=False)
    return nodal_terms, modal_terms


def _make_filter_fn(scaling):
    rescale = lambda x: scaling * x if np.shape(x) == np.broadcast_shapes(
        np.shape(x), scaling.shape) else x
    return functools.partial(jax.tree_util.tree_map, rescale)


def explicit_terms(state):
    vort = inverse_transform(state.vorticity)
    div = inverse_transform(state.divergence)
    temp = inverse_transform(state.temperature_variation)
    tracers = to_nodal(state.tracers)
    l = np.arange(1, g.total_wavenumbers)
    inverse_eigenvalues = np.zeros(g.total_wavenumbers)
    inverse_eigenvalues[1:] = -1 / (l * (l + 1))
    stream_function = state.vorticity * inverse_eigenvalues
    velocity_potential = state.divergence * inverse_eigenvalues

    c00 = real_basis_derivative(velocity_potential)
    c01 = cos_lat_d_dlat(velocity_potential)
    c10 = real_basis_derivative(stream_function)
    c11 = cos_lat_d_dlat(stream_function)
    v0 = c00 - c11
    v1 = c01 + c10
    u = inverse_transform(v0)
    v = inverse_transform(v1)
    grad_u = inverse_transform(
        real_basis_derivative(state.log_surface_pressure))
    grad_v = inverse_transform(cos_lat_d_dlat(state.log_surface_pressure))
    sin_lat, _ = scipy.special.roots_legendre(g.latitude_nodes)
    sec2 = 1 / (1 - sin_lat**2)
    u_dot_grad = u * grad_u * sec2 + v * grad_v * sec2
    f_exp = cumulative_sigma_integral(u_dot_grad)
    f_full = cumulative_sigma_integral(div + u_dot_grad)
    sum_sigma = np.cumsum(g.layer_thickness)[:, None, None]
    sigma_dot = lambda f: jax.lax.slice_in_dim(
        sum_sigma * jax.lax.slice_in_dim(f, -1, None) - f, 0, -1)
    sigma_exp = sigma_dot(f_exp)
    sigma_full = sigma_dot(f_full)

    _, coriolis = np.meshgrid(*nodal_axes(), indexing="ij")
    total_vort = vort + coriolis
    vort_u = -v * total_vort * sec2
    vort_v = u * total_vort * sec2
    sigma_u = -centered_vertical_advection(sigma_full, u)
    sigma_v = -centered_vertical_advection(sigma_full, v)
    rt = ideal_gas_constant * temp
    vert_u = (sigma_u + rt * grad_u) * sec2
    vert_v = (sigma_v + rt * grad_v) * sec2
    u_mod = to_modal(vort_u + vert_u)
    v_mod = to_modal(vort_v + vert_v)

    vort_tendency = -curl_cos_lat((u_mod, v_mod), clip=False)
    div_tendency = -div_cos_lat((u_mod, v_mod), clip=False)

    ke = jnp.stack((u, v))**2
    ke = ke.sum(0) * sec2 / 2
    ke_tendency = -laplacian(to_modal(ke))
    oro_tendency = -gravity_acceleration * laplacian(g.orography)

    h_adv = functools.partial(horizontal_scalar_advection,
                              cos_lat_u=(u, v),
                              divergence=div)
    temp_h_nodal, temp_h_modal = h_adv(temp)
    tracers_h = jax.tree_util.tree_map(h_adv, tracers)

    temp_vert = centered_vertical_advection(sigma_full, temp)
    if np.unique(g.reference_temperature[..., None, None].ravel()).size > 1:
        temp_vert += centered_vertical_advection(
            sigma_exp, g.reference_temperature[..., None, None])

    t_mean = _t_omega_over_sigma_sp(g.reference_temperature[..., None, None],
                                    u_dot_grad, u_dot_grad)
    t_var = _t_omega_over_sigma_sp(temp, div + u_dot_grad, u_dot_grad)
    temp_adiab = kappa * (t_mean + t_var)

    logsp_tendency = -sigma_integral(u_dot_grad)
    tracers_v = jax.tree_util.tree_map(
        lambda x: centered_vertical_advection(sigma_full, x), tracers)

    return clip_wavenumbers(
        State(
            vort_tendency, div_tendency + ke_tendency + oro_tendency,
            to_modal(temp_h_nodal + temp_vert + temp_adiab) + temp_h_modal,
            to_modal(logsp_tendency),
            jax.tree_util.tree_map(
                lambda vert, pair: to_modal(vert + pair[0]) + pair[1],
                tracers_v, tracers_h)))


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
    return State(vorticity_implicit, divergence_implicit,
                 temperature_variation_implicit, log_surface_pressure_implicit,
                 tracers_implicit)


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
    return State(state.vorticity, inverted_divergence,
                 inverted_temperature_variation, inverted_log_surface_pressure,
                 state.tracers)
