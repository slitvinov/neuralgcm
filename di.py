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


def transform(x):
    wx = g.w * x
    fwx = einsum("im,...ij->...mj", g.f, wx)
    pfwx = einsum("mjl,...mj->...ml", g.p, fwx)
    return pfwx


def inverse_transform(x):
    return einsum("im,mjl,...ml->...ij", g.f, g.p, x)


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


def shift_p1(z):
    y = jax.lax.slice_in_dim(z, 0, g.total_wavenumbers - 1, axis=-1)
    return jax.lax.pad(y, 0.0, ((0, 0, 0), (0, 0, 0), (1, 0, 0)))

def shift_m1(z):
    y = jax.lax.slice_in_dim(z, 1, g.total_wavenumbers, axis=-1)
    return jax.lax.pad(y, 0.0, ((0, 0, 0), (0, 0, 0), (0, 1, 0)))


def sigma_integral(x):
    x_axes = range(x.ndim)
    axes = [x_axes[-3]]
    xds = einsum(x, x_axes, g.layer_thickness, axes, x_axes)
    return xds.sum(axis=-3, keepdims=True)


def centered_vertical_advection(w, x):
    w_slc_shape = 1, g.longitude_nodes, g.latitude_nodes
    x_slc_shape = list(x.shape)
    x_slc_shape[-3] = 1
    w_boundary_top = jnp.zeros(w_slc_shape)
    w_boundary_bot = jnp.zeros(w_slc_shape)
    w = jnp.concatenate([w_boundary_top, w, w_boundary_bot], axis=-3)
    dx = jax.lax.slice_in_dim(x, 1, None, axis=-3) - jax.lax.slice_in_dim(
        x, 0, -1, axis=-3)
    inv_ds = 1 / g.center_to_center
    x_diff = einsum(dx, [0, 1, 2], inv_ds, [0], [0, 1, 2], precision="float32")
    x_diff_boundary_top = jnp.zeros(x_slc_shape)
    x_diff_boundary_bot = jnp.zeros(x_slc_shape)
    x_diff = jnp.concatenate(
        [x_diff_boundary_top, x_diff, x_diff_boundary_bot], axis=-3)
    w_times_x_diff = w * x_diff
    return -0.5 * (jax.lax.slice_in_dim(w_times_x_diff, 1, None, axis=-3) +
                   jax.lax.slice_in_dim(w_times_x_diff, 0, -1, axis=-3))


def real_basis_derivative(u):
    n = 2 * g.longitude_wavenumbers - 1
    i = jnp.arange(n).reshape(-1, 1)
    y = jax.lax.slice_in_dim(u, 1, n, axis=-2)
    z = jax.lax.slice_in_dim(u, 0, n - 1, axis=-2)
    u_do = jax.lax.pad(y, 0.0, ((0, 0, 0), (0, 1, 0), (0, 0, 0)))
    u_up = jax.lax.pad(z, 0.0, ((0, 0, 0), (1, 0, 0), (0, 0, 0)))
    return (i + 1) // 2 * jnp.where(i % 2, u_do, -u_up)


def derivative_recurrence_weights():
    p = np.arange(1, g.longitude_wavenumbers)
    q = np.stack([p, -p], axis=1).ravel()
    m, l = np.meshgrid(np.concatenate([[0], q]),
                       np.arange(g.total_wavenumbers),
                       indexing="ij")
    mask = abs(m) <= l
    a = np.sqrt(mask * (l**2 - m**2) / (4 * l**2 - 1))
    a[:, 0] = 0
    b = np.sqrt(mask * ((l + 1)**2 - m**2) / (4 * (l + 1)**2 - 1))
    b[:, -1] = 0
    return a, b


def cos_lat_d_dlat(x):
    l0 = np.arange(g.total_wavenumbers)
    l = np.tile(l0, (2 * g.longitude_wavenumbers - 1, 1))
    a, b = derivative_recurrence_weights()
    x_lm1 = shift_m1(((l + 1) * a) * x)
    x_lp1 = shift_p1((-l * b) * x)
    return x_lm1 + x_lp1


def sec_lat_d_dlat_cos2(x):
    l0 = np.arange(g.total_wavenumbers)
    l = np.tile(l0, (2 * g.longitude_wavenumbers - 1, 1))
    a, b = derivative_recurrence_weights()
    x_lm1 = shift_m1(((l - 1) * a) * x)
    x_lp1 = shift_p1((-(l + 2) * b) * x)
    return x_lm1 + x_lp1


def runge_kutta(exp, imp, inv, dt):
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
    vo: Any
    di: Any
    te: Any
    sp: Any
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


def _t_omega_over_sigma_sp(temperature_field, g_term, v_dot_grad_log_sp):
    f = jax.lax.cumsum(g_term * g.layer_thickness[:, None, None])
    alpha = get_sigma_ratios()
    alpha = alpha[:, np.newaxis, np.newaxis]
    del_s = g.layer_thickness[:, np.newaxis, np.newaxis]
    padding = [(1, 0), (0, 0), (0, 0)]
    g_part = (alpha * f + jnp.pad(alpha * f, padding)[:-1, ...]) / del_s
    return temperature_field * (v_dot_grad_log_sp - g_part)


def horizontal_scalar_advection(scalar, cos_lat_u, divergence):
    u, v = cos_lat_u
    nodal_terms = scalar * divergence
    sin_lat, _ = scipy.special.roots_legendre(g.latitude_nodes)
    sec2 = 1 / (1 - sin_lat**2)
    m_component = transform(u * scalar * sec2)
    n_component = transform(v * scalar * sec2)
    modal_terms = -real_basis_derivative(m_component) - sec_lat_d_dlat_cos2(
        n_component)
    return nodal_terms, modal_terms


def _make_filter_fn(scaling):
    rescale = lambda x: scaling * x if np.shape(x) == np.broadcast_shapes(
        np.shape(x), scaling.shape) else x
    return functools.partial(jax.tree_util.tree_map, rescale)


def explicit_terms(s):
    vort = inverse_transform(s.vo)
    div = inverse_transform(s.di)
    temp = inverse_transform(s.te)
    tracers = tree_map(inverse_transform, s.tracers)
    l = np.arange(1, g.total_wavenumbers)
    inverse_eigenvalues = np.zeros(g.total_wavenumbers)
    inverse_eigenvalues[1:] = -1 / (l * (l + 1))
    stream_function = s.vo * inverse_eigenvalues
    velocity_potential = s.di * inverse_eigenvalues

    c00 = real_basis_derivative(velocity_potential)
    c01 = cos_lat_d_dlat(velocity_potential)
    c10 = real_basis_derivative(stream_function)
    c11 = cos_lat_d_dlat(stream_function)
    v0 = c00 - c11
    v1 = c01 + c10
    u = inverse_transform(v0)
    v = inverse_transform(v1)
    grad_u = inverse_transform(real_basis_derivative(s.sp))
    grad_v = inverse_transform(cos_lat_d_dlat(s.sp))
    sin_lat, _ = scipy.special.roots_legendre(g.latitude_nodes)
    sec2 = 1 / (1 - sin_lat**2)
    u_dot_grad = u * grad_u * sec2 + v * grad_v * sec2
    f_exp = jax.lax.cumsum(u_dot_grad * g.layer_thickness[:, None, None])
    f_full = jax.lax.cumsum(
        (div + u_dot_grad) * g.layer_thickness[:, None, None])
    sum_sigma = np.cumsum(g.layer_thickness)[:, None, None]
    sigma_dot = lambda f: jax.lax.slice_in_dim(
        sum_sigma * jax.lax.slice_in_dim(f, -1, None) - f, 0, -1)
    sigma_exp = sigma_dot(f_exp)
    sigma_full = sigma_dot(f_full)
    sin_latitude, _ = scipy.special.roots_legendre(g.latitude_nodes)
    coriolis = np.tile(sin_latitude, (g.longitude_nodes, 1))
    total_vort = vort + coriolis
    vort_u = -v * total_vort * sec2
    vort_v = u * total_vort * sec2
    sigma_u = -centered_vertical_advection(sigma_full, u)
    sigma_v = -centered_vertical_advection(sigma_full, v)
    rt = ideal_gas_constant * temp
    vert_u = (sigma_u + rt * grad_u) * sec2
    vert_v = (sigma_v + rt * grad_v) * sec2
    u_mod = transform(vort_u + vert_u)
    v_mod = transform(vort_v + vert_v)

    vort_tendency = -real_basis_derivative(v_mod) + sec_lat_d_dlat_cos2(u_mod)
    div_tendency = -real_basis_derivative(u_mod) - sec_lat_d_dlat_cos2(v_mod)

    ke = jnp.stack((u, v))**2
    ke = ke.sum(0) * sec2 / 2
    l0 = np.arange(g.total_wavenumbers)
    ke_tendency = l0 * (l0 + 1) * transform(ke)
    oro_tendency = gravity_acceleration * (l0 * (l0 + 1) * g.orography)

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
    mask = jnp.ones(g.total_wavenumbers).at[-1:].set(0)
    return State(
        vort_tendency * mask,
        (div_tendency + ke_tendency + oro_tendency) * mask,
        (transform(temp_h_nodal + temp_vert + temp_adiab) + temp_h_modal) *
        mask,
        transform(logsp_tendency) * mask,
        jax.tree_util.tree_map(
            lambda vert, pair: (transform(vert + pair[0]) + pair[1]) * mask,
            tracers_v, tracers_h))


def implicit_terms(s):
    weights = get_geopotential_weights()
    geopotential_diff = einsum("gh,...hml->...gml", weights, s.te)
    rt_log_p = (ideal_gas_constant *
                g.reference_temperature[..., np.newaxis, np.newaxis] * s.sp)
    vorticity_implicit = jnp.zeros_like(s.vo)
    l0 = np.arange(g.total_wavenumbers)
    divergence_implicit = l0 * (l0 + 1) * (geopotential_diff + rt_log_p)
    weights = -get_temperature_implicit_weights()
    temperature_variation_implicit = einsum("gh,...hml->...gml", weights, s.di)
    log_surface_pressure_implicit = -einsum(
        "gh,...hml->...gml", g.layer_thickness[np.newaxis], s.di)
    tracers_implicit = jax.tree_util.tree_map(jnp.zeros_like, s.tracers)
    return State(vorticity_implicit, divergence_implicit,
                 temperature_variation_implicit, log_surface_pressure_implicit,
                 tracers_implicit)


def implicit_inverse(s, dt):
    eye = np.eye(g.layers)[np.newaxis]
    l0 = np.arange(g.total_wavenumbers)
    lam = -l0 * (l0 + 1)
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
    inv = np.linalg.inv(implicit_matrix)
    div = slice(0, g.layers)
    temp = slice(g.layers, 2 * g.layers)
    logp = slice(2 * g.layers, 2 * g.layers + 1)
    inverted_divergence = (
        einsum("lgh,...hml->...gml", inv[:, div, div], s.di) +
        einsum("lgh,...hml->...gml", inv[:, div, temp], s.te) +
        einsum("lgh,...hml->...gml", inv[:, div, logp], s.sp))
    inverted_temperature_variation = (
        einsum("lgh,...hml->...gml", inv[:, temp, div], s.di) +
        einsum("lgh,...hml->...gml", inv[:, temp, temp], s.te) +
        einsum("lgh,...hml->...gml", inv[:, temp, logp], s.sp))
    inverted_log_surface_pressure = (
        einsum("lgh,...hml->...gml", inv[:, logp, div], s.di) +
        einsum("lgh,...hml->...gml", inv[:, logp, temp], s.te) +
        einsum("lgh,...hml->...gml", inv[:, logp, logp], s.sp))
    return State(s.vo, inverted_divergence, inverted_temperature_variation,
                 inverted_log_surface_pressure, s.tracers)
