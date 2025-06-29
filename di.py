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
r_gas = kappa * 0.0011628807950492582


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
    q = np.sqrt(1 - x * x)
    y = np.zeros(
        (g.total_wavenumbers, g.longitude_wavenumbers, g.latitude_nodes))
    y[0, 0] = 1 / np.sqrt(2)
    for m in range(1, g.longitude_wavenumbers):
        y[0, m] = -np.sqrt(1 + 1 / (2 * m)) * q * y[0, m - 1]
    for k in range(1, g.total_wavenumbers):
        M = min(g.longitude_wavenumbers, g.total_wavenumbers - k)
        m = np.c_[:M]
        m2 = m**2
        mk2 = (m + k)**2
        mkp2 = (m + k - 1)**2
        a = np.sqrt((4 * mk2 - 1) / (mk2 - m2))
        b = np.sqrt((mkp2 - m2) / (4 * mkp2 - 1))
        y[k, :M] = a * (x * y[k - 1, :M] - b * y[k - 2, :M])
    r = np.transpose(y, (1, 2, 0))
    p = np.zeros(
        (g.longitude_wavenumbers, g.latitude_nodes, g.total_wavenumbers))
    for m in range(g.longitude_wavenumbers):
        p[m, :, m:g.total_wavenumbers] = r[m, :, 0:g.total_wavenumbers - m]
    p = np.repeat(p, 2, axis=0)
    return f, p[1:], 2 * np.pi * w / g.longitude_nodes


def sigma_integral(x):
    x_axes = range(x.ndim)
    axes = [x_axes[-3]]
    xds = einsum(x, x_axes, g.thick, axes, x_axes)
    return xds.sum(axis=-3, keepdims=True)


def vadvection(w, x):
    shape = list(x.shape)
    shape[-3] = 1
    wt = np.zeros((1, g.longitude_nodes, g.latitude_nodes))
    xt = np.zeros(shape)
    dx = x[1:] - x[:-1]
    xd = einsum(dx, [0, 1, 2], 1 / g.center_to_center, [0], [0, 1, 2])
    wx = jnp.concatenate([wt, w, wt]) * jnp.concatenate([xt, xd, xt])
    return -0.5 * (wx[1:] + wx[:-1])


def real_basis_derivative(u):
    n = 2 * g.longitude_wavenumbers - 1
    y = u[:, 1:n, :]
    z = u[:, :n - 1, :]
    u_do = jax.lax.pad(y, 0.0, ((0, 0, 0), (0, 1, 0), (0, 0, 0)))
    u_up = jax.lax.pad(z, 0.0, ((0, 0, 0), (1, 0, 0), (0, 0, 0)))
    i = np.c_[:n]
    return (i + 1) // 2 * jnp.where(i % 2, u_do, -u_up)


def derivative_recurrence_weights():
    p = np.r_[1:g.longitude_wavenumbers]
    q = np.c_[p, -p]
    m, l = np.meshgrid(np.r_[0, q.ravel()],
                       np.r_[:g.total_wavenumbers],
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
    zm = (l + 1) * a * x
    zp = -l * b * x
    lm1 = jax.lax.pad(zm[:, :, 1:g.total_wavenumbers], 0.0,
                      ((0, 0, 0), (0, 0, 0), (0, 1, 0)))
    lp1 = jax.lax.pad(zp[:, :, :g.total_wavenumbers - 1], 0.0,
                      ((0, 0, 0), (0, 0, 0), (1, 0, 0)))
    return lm1 + lp1


def sec_lat_d_dlat_cos2(x):
    l0 = np.arange(g.total_wavenumbers)
    l = np.tile(l0, (2 * g.longitude_wavenumbers - 1, 1))
    a, b = derivative_recurrence_weights()
    zm = (l - 1) * a * x
    zp = -(l + 2) * b * x
    lm1 = jax.lax.pad(zm[:, :, 1:g.total_wavenumbers], 0.0,
                      ((0, 0, 0), (0, 0, 0), (0, 1, 0)))
    lp1 = jax.lax.pad(zp[:, :, :g.total_wavenumbers - 1], 0.0,
                      ((0, 0, 0), (0, 0, 0), (1, 0, 0)))
    return lm1 + lp1


def runge_kutta(exp, imp, inv, dt):
    F = tree_math.unwrap(exp)
    G = tree_math.unwrap(imp)
    G_inv = tree_math.unwrap(inv, vector_argnums=0)
    a_ex = [1 / 3], [1 / 6, 1 / 2], [1 / 2, -1 / 2, 1]
    a_im = [1 / 6, 1 / 6], [1 / 3, 0, 1 / 3], [3 / 8, 0, 3 / 8, 1 / 4]
    b_ex = 1 / 2, -1 / 2, 1, 0
    b_im = 3 / 8, 0, 3 / 8, 1 / 4
    n = len(b_ex)

    @tree_math.wrap
    def step_fn(y):
        f = [None] * n
        g = [None] * n
        f[0] = F(y)
        g[0] = G(y)
        for i in range(1, n):
            ex = dt * sum(a_ex[i - 1][j] * f[j]
                          for j in range(i) if a_ex[i - 1][j])
            im = dt * sum(a_im[i - 1][j] * g[j]
                          for j in range(i) if a_im[i - 1][j])
            Y = G_inv(y + ex + im, dt * a_im[i - 1][i])
            if any(a_ex[j][i] for j in range(i, n - 1)) or b_ex[i]: f[i] = F(Y)
            if any(a_im[j][i] for j in range(i, n - 1)) or b_im[i]: g[i] = G(Y)
        ex = dt * sum(b_ex[j] * f[j] for j in range(n) if b_ex[j])
        im = dt * sum(b_im[j] * g[j] for j in range(n) if b_im[j])
        return y + ex + im

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


def geopotential_weights():
    alpha = get_sigma_ratios()
    weights = np.zeros([g.layers, g.layers])
    for j in range(g.layers):
        weights[j, j] = alpha[j]
        for k in range(j + 1, g.layers):
            weights[j, k] = alpha[k] + alpha[k - 1]
    return r_gas * weights


def get_temperature_implicit_weights():
    p = np.tril(np.ones([g.layers, g.layers]))
    alpha = get_sigma_ratios()[..., None]
    p_alpha = p * alpha
    p_alpha_shifted = np.roll(p_alpha, 1, axis=0)
    p_alpha_shifted[0] = 0
    h0 = (kappa * g.temp[..., None] * (p_alpha + p_alpha_shifted) /
          g.thick[..., None])
    temp_diff = np.diff(g.temp)
    thickness_sum = g.thick[:-1] + g.thick[1:]
    k0 = np.concatenate((temp_diff / thickness_sum, [0]), axis=0)[..., None]
    thickness_cumulative = np.cumsum(g.thick)[..., None]
    k1 = p - thickness_cumulative
    k = k0 * k1
    k_shifted = np.roll(k, 1, axis=0)
    k_shifted[0] = 0
    return (h0 - k - k_shifted) * g.thick


def omega(g_term):
    f = jax.lax.cumsum(g_term * g.thick[:, None, None])
    alpha = get_sigma_ratios()[:, None, None]
    pad = (1, 0), (0, 0), (0, 0)
    return (alpha * f + jnp.pad(alpha * f, pad)[:-1, ...]) / g.thick[:, None,
                                                                     None]


def hadvection(scalar, cos_lat_u, divergence):
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
    f_exp = jax.lax.cumsum(u_dot_grad * g.thick[:, None, None])
    f_full = jax.lax.cumsum((div + u_dot_grad) * g.thick[:, None, None])
    sum_sigma = np.cumsum(g.thick)[:, None, None]
    sigma_exp = (sum_sigma * f_exp[-1] - f_exp)[:-1]
    sigma_full = (sum_sigma * f_full[-1] - f_full)[:-1]
    sin_latitude, _ = scipy.special.roots_legendre(g.latitude_nodes)
    coriolis = np.tile(sin_latitude, (g.longitude_nodes, 1))
    total_vort = vort + coriolis
    vort_u = -v * total_vort * sec2
    vort_v = u * total_vort * sec2
    sigma_u = -vadvection(sigma_full, u)
    sigma_v = -vadvection(sigma_full, v)
    rt = r_gas * temp
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

    h_adv = functools.partial(hadvection, cos_lat_u=(u, v), divergence=div)
    temp_h_nodal, temp_h_modal = h_adv(temp)
    tracers_h = jax.tree_util.tree_map(h_adv, tracers)

    temp_vert = vadvection(sigma_full, temp)
    if np.unique(g.temp[..., None, None].ravel()).size > 1:
        temp_vert += vadvection(sigma_exp, g.temp[..., None, None])

    t_mean = g.temp[..., None, None] * (u_dot_grad - omega(u_dot_grad))
    t_var = temp * (u_dot_grad - omega(div + u_dot_grad))
    temp_adiab = kappa * (t_mean + t_var)

    logsp_tendency = -sigma_integral(u_dot_grad)
    tracers_v = jax.tree_util.tree_map(lambda x: vadvection(sigma_full, x),
                                       tracers)
    mask = np.r_[[1] * (g.total_wavenumbers - 1), 0]
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
    geopotential_diff = einsum("gh,hml->gml", g.geo, s.te)
    rt_log_p = r_gas * g.temp[..., None, None] * s.sp
    vorticity_implicit = jnp.zeros_like(s.vo)
    l0 = np.arange(g.total_wavenumbers)
    divergence_implicit = l0 * (l0 + 1) * (geopotential_diff + rt_log_p)
    weights = -get_temperature_implicit_weights()
    temperature_variation_implicit = einsum("gh,hml->gml", weights, s.di)
    log_surface_pressure_implicit = -einsum("gh,hml->gml", g.thick[None], s.di)
    tracers_implicit = jax.tree_util.tree_map(jnp.zeros_like, s.tracers)
    return State(vorticity_implicit, divergence_implicit,
                 temperature_variation_implicit, log_surface_pressure_implicit,
                 tracers_implicit)


def implicit_inverse(s, dt):
    eye = np.eye(g.layers)
    l0 = np.arange(g.total_wavenumbers)
    lam = -l0 * (l0 + 1)
    h = get_temperature_implicit_weights()[None]
    l = g.total_wavenumbers
    j = k = g.layers
    row0 = np.c_[np.broadcast_to(eye, [l, j, k]),
                 dt * np.einsum("l,jk->ljk", lam, g.geo),
                 dt * r_gas * np.einsum("l,jo->ljo", lam, g.temp[:, None])]
    row1 = np.c_[dt * np.broadcast_to(h, [l, j, k]),
                 np.broadcast_to(eye, [l, j, k]),
                 np.zeros([l, j, 1])]
    row2 = np.c_[dt * np.broadcast_to(g.thick, [l, 1, k]),
                 np.zeros([l, 1, k]),
                 np.ones([l, 1, 1])]
    inv = np.linalg.inv(np.r_['1', row0, row1, row2])
    div = np.s_[:j]
    temp = np.s_[j:2 * j]
    logp = np.s_[2 * j:2 * j + 1]
    di = (einsum("lgh,hml->gml", inv[:, div, div], s.di) +
          einsum("lgh,hml->gml", inv[:, div, temp], s.te) +
          einsum("lgh,hml->gml", inv[:, div, logp], s.sp))
    te = (einsum("lgh,hml->gml", inv[:, temp, div], s.di) +
          einsum("lgh,hml->gml", inv[:, temp, temp], s.te) +
          einsum("lgh,hml->gml", inv[:, temp, logp], s.sp))
    sp = (einsum("lgh,hml->gml", inv[:, logp, div], s.di) +
          einsum("lgh,hml->gml", inv[:, logp, temp], s.te) +
          einsum("lgh,hml->gml", inv[:, logp, logp], s.sp))
    return State(s.vo, di, te, sp, s.tracers)
