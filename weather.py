import functools
import jax
import jax.numpy as jnp
import math
import numpy as np
import os
import scipy
import xarray


class g:
    pass


@jax.jit
def step(s):
    return jax.lax.fori_loop(0, g.inner,
                             lambda _, x: g.scale * runge_kutta(1, x), s)


def accumulate(sign):

    def f(carry, w):
        s, a = carry
        s = g.scale * runge_kutta(sign, s)
        a += w * s
        return (s, a), None

    (_, a), _ = jax.lax.scan(f, (s, g.zero), g.weights)
    return a


def roll(a, shift):
    for ax, s in enumerate(shift):
        if a.shape[ax] == 0:
            continue
        i = (-s) % a.shape[ax]
        p, q = a, jnp.zeros_like(a)
        if s > 0:
            p, q = q, p
        a = jax.lax.concatenate([
            jax.lax.slice_in_dim(p, i, a.shape[ax], axis=ax),
            jax.lax.slice_in_dim(q, 0, i, axis=ax)
        ],
                                dimension=ax)

    return a


def modal_fft(x):
    nz, *rest = np.shape(x)
    s0 = 1 / math.sqrt(math.pi) / math.sqrt(2)
    s1 = 1 / math.sqrt(math.pi)
    u = jnp.fft.rfft(g.w * x, axis=1)
    a0 = s0 * u[:, :1, :].real
    u0 = s1 * u[:, 1:g.m, :].real
    u1 = -s1 * u[:, 1:g.m, :].imag
    a1 = jnp.r_['2', u0, u1].reshape((nz, -1, g.ny))
    a = jnp.r_['1', a0, a1]
    return einsum("mjl,...mj->...ml", g.p, a)


def modal_direct(x):
    wx = g.w * x
    fwx = einsum("im,...ij->...mj", g.f, wx)
    return einsum("mjl,...mj->...ml", g.p, fwx)


def nodal_fft(x):
    nz, *rest = np.shape(x)
    s0 = math.sqrt(math.pi) * math.sqrt(2)
    s1 = math.sqrt(math.pi)
    s2 = g.nx / s0**2
    u = einsum("mjl,...ml->...mj", g.p, x)
    F0 = u[:, :1, :] * s0
    F1 = (u[:, 1::2, :] - 1j * u[:, 2::2, :]) * s1
    assert g.nx // 2 + 1 - g.m >= 0, "g.nx is too small"
    Fpad = jnp.zeros((nz, g.nx // 2 + 1 - g.m, g.ny))
    F = jnp.r_['1', F0, F1, Fpad]
    return jnp.fft.irfft(F, axis=1) * s2


def nodal_direct(x):
    return einsum("im,mjl,...ml->...ij", g.f, g.p, x)


def dx(u):
    lo = roll(u, [0, -1, 0])
    hi = roll(u, [0, 1, 0])
    i = np.c_[:2 * g.m - 1]
    return (i + 1) // 2 * jnp.where(i % 2, lo, -hi)


def dy(x):
    return pad(g.ax * x, g.bx * x)


def dy_cos(x):
    return pad(g.ay * x, g.by * x)


def pad(a, b):
    return roll(a, [0, 0, -1]) + roll(b, [0, 0, 1])


def runge_kutta(sign, y):
    n = len(g.b_ex)
    f = [None] * n
    h = [None] * n
    f[0] = sign * F(y)
    h[0] = sign * G(y)
    for i in range(1, n):
        ex = g.dt * sum(g.a_ex[i - 1][j] * f[j]
                        for j in range(i) if g.a_ex[i - 1][j])
        im = g.dt * sum(g.a_im[i - 1][j] * h[j]
                        for j in range(i) if g.a_im[i - 1][j])
        tau = g.dt * g.a_im[i - 1][i]
        Y = G_inv(y + ex + im, sign * tau)
        if any(g.a_ex[j][i] for j in range(i, n - 1)) or g.b_ex[i]:
            f[i] = sign * F(Y)
        if any(g.a_im[j][i] for j in range(i, n - 1)) or g.b_im[i]:
            h[i] = sign * G(Y)
    ex = g.dt * sum(g.b_ex[j] * f[j] for j in range(n) if g.b_ex[j])
    im = g.dt * sum(g.b_im[j] * h[j] for j in range(n) if g.b_im[j])
    return y + ex + im


def F(s):

    def hadv(x):
        return -dx(modal(u * x * g.sec2)) - dy(modal(v * x * g.sec2))

    def vadv(x):
        wt = np.zeros((1, g.nx, g.ny))
        dx = x[1:] - x[:-1]
        xd = dx * g.nz
        wx = jnp.r_[wt, xd * dot_sigma, wt]
        return -0.5 * (wx[1:] + wx[:-1])

    def omega(x):
        f = jax.lax.cumsum(x)
        alpha = g.alpha[:, None, None]
        return alpha * f + roll(alpha * f, [1, 0, 0])

    nod = nodal(s)
    vo, di, te, hu, wo, ic = nod[g.vo], nod[g.di], nod[g.te], nod[g.hu], nod[
        g.wo], nod[g.ic]

    tv = te * (g.eps * hu + 1)

    psi = s[g.vo] * g.inv_eig
    chi = s[g.di] * g.inv_eig

    dchi_dx = dx(chi)
    dchi_dy_cos = dy_cos(chi)
    dpsi_dx = dx(psi)
    dpsi_dy_cos = dy_cos(psi)

    u = nodal(dchi_dx - dpsi_dy_cos)
    v = nodal(dchi_dy_cos + dpsi_dx)

    spx = nodal(dx(s[g.sp]))
    spy = nodal(dy_cos(s[g.sp]))

    u_dot_grad_sp = (u * spx + v * spy) * g.sec2

    int_div = jax.lax.cumsum(di + u_dot_grad_sp) / g.nz
    dot_sigma = (g.sigma[:, None, None] * int_div[-1] - int_div)[:-1]

    abs_vo = vo + g.sin_y[None, None, :]  # coriolis

    fvx = -v * abs_vo * g.sec2
    fvy = u * abs_vo * g.sec2

    vadv_u = -vadv(u)
    vadv_v = -vadv(v)

    RT = g.r_dry * tv
    sp_force_x = RT * spx
    sp_force_y = RT * spy

    fx = fvx + (vadv_u + sp_force_x) * g.sec2
    fy = fvy + (vadv_v + sp_force_y) * g.sec2

    fx_spec = modal(fx)
    fy_spec = modal(fy)

    dvo = -dx(fy_spec) + dy(fx_spec)
    ddi = -dx(fx_spec) - dy(fy_spec)

    ke = g.sec2 * (u**2 + v**2)
    dke = g.eig * modal(ke)
    ddi += 0.5 * dke + g.doro

    dte_hadv = hadv(te)
    dte_vadv = vadv(te)

    omega_mean = omega(u_dot_grad_sp)
    omega_full = omega(di + u_dot_grad_sp)

    tempv = g.temp * (g.eps * hu + 1)
    dte_adiab = kappa * (tempv * (u_dot_grad_sp - omega_mean) + tv *
                         (u_dot_grad_sp - omega_full))
    dte = modal(te * di + dte_vadv + dte_adiab) + dte_hadv

    dhu_hadv = hadv(hu)
    dwo_hadv = hadv(wo)
    dic_hadv = hadv(ic)

    dhu_vadv = vadv(hu)
    dwo_vadv = vadv(wo)
    dic_vadv = vadv(ic)

    dhu_dil = hu * di
    dwo_dil = wo * di
    dic_dil = ic * di

    dmoist_vadv = jnp.r_[dhu_vadv, dwo_vadv, dic_vadv]
    dmoist_dil = jnp.r_[dhu_dil, dwo_dil, dic_dil]
    dmoist_hadv = jnp.r_[dhu_hadv, dwo_hadv, dic_hadv]
    dmoist = modal(dmoist_vadv + dmoist_dil) + dmoist_hadv

    dsp_phys = -jnp.sum(u_dot_grad_sp, axis=0, keepdims=True) / g.nz
    dsp = modal(dsp_phys)
    return jnp.r_[dvo, ddi, dte, dsp, dmoist] * g.mask


def G(s):
    shape = g.nz, 2 * g.m - 1, g.l
    tscale = 3 * g.nz, 2 * g.m - 1, g.l
    ddi = g.eig * (einsum("gh,hml->gml", g.geo, s[g.te]) +
                   g.r_dry * g.temp * s[g.sp])
    T = np.full((1, g.nz), -1 / g.nz)
    dtesp = einsum("gh,hml->gml", jnp.r_[-g.tew, T], s[g.di])
    return jnp.r_[jnp.zeros(shape), ddi, dtesp, jnp.zeros(tscale)]


def G_inv(s, dt):
    I = np.r_[[np.eye(g.nz)] * g.l]
    A = -dt * g.eig[:, None, None] * g.geo[None]
    B = np.r_[[-dt * g.r_dry * g.eig * g.temp] * g.nz].T[:, :, None]
    C = dt * np.r_[[g.tew] * g.l]
    D = np.full((g.l, 1, g.nz), dt / g.nz)
    Z = np.zeros([g.l, g.nz, 1])
    Z0 = np.zeros([g.l, 1, g.nz])
    I0 = np.ones([g.l, 1, 1])
    row0 = np.c_[I, A, B]
    row1 = np.c_[C, I, Z]
    row2 = np.c_[D, Z0, I0]
    inv = np.linalg.inv(np.r_['1', row0, row1, row2])
    sol = einsum("lgh,hml->gml", inv, s[g.ditesp])
    return jnp.r_[s[g.vo], sol, s[g.hu], s[g.wo], s[g.ic]]


def open(path):
    x = xarray.open_zarr(path, chunks=None, storage_options=dict(token="anon"))
    return x.sel(time="19900501T00")


def interpn(data):
    return scipy.interpolate.interpn(xy_src,
                                     data,
                                     xy_grid,
                                     bounds_error=False,
                                     fill_value=None)


GRAVITY = 9.80616
uL = 6.37122e6
uT = 1 / 2 / 7.292e-5
einsum = functools.partial(jnp.einsum, precision=jax.lax.Precision.HIGHEST)
gravity = GRAVITY / (uL / uT**2)
kappa = 2 / 7
g.temp = 250
g.r_dry = kappa * 1004 * uT**2 / uL**2
r_vap = 461.0 * uT**2 / uL**2
g.eps = r_vap / g.r_dry - 1
g.m = 171
g.l = g.m + 1
g.nx = 3 * g.m + 1
g.ny = g.nx // 2
g.nz = 32
g.zb = np.linspace(0, 1, g.nz + 1)
dft = scipy.linalg.dft(g.nx)[:, :g.m] / math.sqrt(math.pi)
g.f = np.empty((g.nx, 2 * g.m - 1))
g.f[:, 0] = 1 / math.sqrt(2 * math.pi)
g.f[:, 1::2] = np.real(dft[:, 1:])
g.f[:, 2::2] = -np.imag(dft[:, 1:])
g.sin_y, w = scipy.special.roots_legendre(g.ny)
g.sec2 = 1 / (1 - g.sin_y**2)
q = np.sqrt(1 - g.sin_y * g.sin_y)
y = np.zeros((g.l, g.m, g.ny))
y[0, 0] = 1 / np.sqrt(2)
for m in range(1, g.m):
    y[0, m] = -np.sqrt(1 + 1 / (2 * m)) * q * y[0, m - 1]
for k in range(1, g.l):
    fields = min(g.m, g.l - k)
    m = np.c_[:fields]
    m2 = m**2
    mk2 = (m + k)**2
    mkp2 = (m + k - 1)**2
    a = np.sqrt((4 * mk2 - 1) / (mk2 - m2))
    b = np.sqrt((mkp2 - m2) / (4 * mkp2 - 1))
    y[k, :fields] = a * (g.sin_y * y[k - 1, :fields] - b * y[k - 2, :fields])
r = np.transpose(y, (1, 2, 0))
p = np.zeros((g.m, g.ny, g.l))
for m in range(g.m):
    p[m, :, m:g.l] = r[m, :, 0:g.l - m]
p = np.repeat(p, 2, axis=0)
g.p = p[1:]
g.w = 2 * math.pi * w / g.nx

p = np.r_[1:g.m]
q = np.c_[p, -p]
m, l = np.meshgrid(np.r_[0, q.ravel()], np.r_[:g.l], indexing="ij")
mask = abs(m) <= l
g.a = np.sqrt(mask * (l**2 - m**2) / (4 * l**2 - 1))
g.a[:, 0] = 0

g.b = np.sqrt(mask * ((l + 1)**2 - m**2) / (4 * (l + 1)**2 - 1))
g.b[:, -1] = 0
zc = (g.zb[1:] + g.zb[:-1]) / 2
g.alpha = np.diff(np.log(zc), append=0) / 2
g.alpha[-1] = -np.log(zc[-1])

weights = np.zeros([g.nz, g.nz])
for j in range(g.nz):
    weights[j, j] = g.alpha[j]
    for k in range(j + 1, g.nz):
        weights[j, k] = g.alpha[k] + g.alpha[k - 1]
g.geo = g.r_dry * weights
p_alpha = np.tril(np.ones([g.nz, g.nz])) * g.alpha[..., None]
p_alpha_shifted = np.roll(p_alpha, 1, axis=0)
p_alpha_shifted[0] = 0
g.tew = kappa * g.temp * (p_alpha + p_alpha_shifted)
g.l0 = np.r_[:g.l]
g.eig = g.l0 * (g.l0 + 1)
g.inv_eig = np.r_[0, -1 / g.eig[1:]]
g.mask = np.r_[[1] * (g.l - 1), 0]
g.sigma = np.linspace(1 / g.nz, 1, g.nz)
g.ax = (g.l0 - 1) * g.a
g.bx = -(g.l0 + 2) * g.b
g.ay = (g.l0 + 1) * g.a
g.by = -g.l0 * g.b
g.a_ex = [1 / 3], [1 / 6, 1 / 2], [1 / 2, -1 / 2, 1]
g.a_im = [1 / 6, 1 / 6], [1 / 3, 0, 1 / 3], [3 / 8, 0, 3 / 8, 1 / 4]
g.b_ex = 1 / 2, -1 / 2, 1, 0
g.b_im = 3 / 8, 0, 3 / 8, 1 / 4
modal = modal_direct
nodal = nodal_direct

n = g.nz
g.vo = np.s_[:n]
g.di = np.s_[n:2 * n]
g.te = np.s_[2 * n:3 * n]
g.sp = np.s_[3 * n:3 * n + 1]
g.hu = np.s_[3 * n + 1:4 * n + 1]
g.wo = np.s_[4 * n + 1:5 * n + 1]
g.ic = np.s_[5 * n + 1:6 * n + 1]
g.ditesp = np.s_[n:3 * n + 1]
shape = 6 * g.nz + 1, 2 * g.m - 1, g.l
if os.path.exists("s.raw") and os.path.exists("doro.raw"):
    s = np.fromfile("s.raw", dtype=np.float32).reshape(shape)
    g.doro = np.fromfile("doro.raw", dtype=np.float32).reshape(shape[1:])
else:
    era = xarray.merge([
        open(
            "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
        ).drop_dims("level"),
        open(
            "gs://gcp-public-data-arco-era5/ar/model-level-1h-0p25deg.zarr-v1")
    ])
    y_deg = np.rad2deg(np.arcsin(g.sin_y))
    x_deg = np.linspace(0, 360, g.nx, endpoint=False)
    a_zb, b_zb = np.fromfile("levels.raw", dtype=np.float32).reshape(2, -1)
    nhyb = len(era["hybrid"].data)
    y_src = era["latitude"].data
    x_src = era["longitude"].data
    xy_grid = np.meshgrid(y_deg, x_deg)
    xy_src = y_src, x_src
    sp = interpn(era["surface_pressure"].data)
    oro = interpn(era["geopotential_at_surface"].data)
    fields = {}
    samples = np.empty((nhyb, g.nx, g.ny))
    source = a_zb[:, None, None] / sp + b_zb[:, None, None]
    upper = np.minimum(g.zb[1:, None, None, None], source[None, 1:, :, :])
    lower = np.maximum(g.zb[:-1, None, None, None], source[None, :-1, :, :])
    weights = np.maximum(upper - lower, 0)
    weights /= np.sum(weights, axis=1, keepdims=True)
    for key, scale in [
        ("u_component_of_wind", uL / uT),
        ("v_component_of_wind", uL / uT),
        ("temperature", 1),
        ("specific_cloud_liquid_water_content", 1),
        ("specific_cloud_ice_water_content", 1),
        ("specific_humidity", 1),
    ]:
        for i in range(nhyb):
            samples[i] = interpn(era[key].data[i])
        fields[key] = np.einsum("lnxy,nxy->lxy", weights, samples / scale)
    cos = np.sqrt(1 - g.sin_y**2)
    u = modal(fields["u_component_of_wind"] / cos)
    v = modal(fields["v_component_of_wind"] / cos)
    vor = dx(v) - dy(u)
    div = dx(u) + dy(v)
    s = np.empty(shape, dtype=np.float32)
    s[g.vo] = vor * g.mask
    s[g.di] = div * g.mask
    s[g.te] = modal(fields["temperature"] - g.temp)
    s[g.sp] = modal(np.log(sp[None, :, :] / (1 / uL / uT**2)))
    s[g.hu] = modal(fields["specific_humidity"])
    s[g.wo] = modal(fields["specific_cloud_liquid_water_content"])
    s[g.ic] = modal(fields["specific_cloud_ice_water_content"])
    k = g.l0 / (g.l - 1)
    g.doro = (gravity * g.eig) * modal(oro[None, :, :]) * np.exp(
        -16 * k**4) / (uL * GRAVITY)
    s.tofile("s.raw")
    np.asarray(oro).tofile("oro.raw")
    np.asarray(g.doro).tofile("doro.raw")

g.dt = 4.3752000000000006e-02
tau = 3600 * 8.6 / (2.4**np.log2(g.ny / 128)) / uT
g.scale = jnp.exp(-g.dt * g.eig**2 / (tau * g.eig[-1]**2))

N = 36
n = np.arange(1, N + 1)
g.weights = np.sinc(n / (N + 1)) * np.sinc(n / N)
norm = 1 + 2 * np.sum(g.weights)
g.weights /= norm

g.zero = np.zeros(shape, dtype=np.float32)
sf = accumulate(1)
sb = accumulate(-1)

np.asarray(s).tofile("out.org.raw")
s /= norm
s += sb
s += sf

g.inner = 50
g.outter = 10
i = 0
while True:
    np.asarray(s).tofile(f"out.{i:08d}.raw")
    if i == g.outter:
        break
    i += 1
    s = step(s)
np.asarray(s).tofile("out.raw")
