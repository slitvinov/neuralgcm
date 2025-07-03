import functools
import jax
import jax.numpy as jnp
import numpy as np
import os
import scipy
import xarray


class g:
    pass


def transform(x):
    wx = g.w * x
    fwx = einsum("im,...ij->...mj", g.f, wx)
    pfwx = einsum("mjl,...mj->...ml", g.p, fwx)
    return pfwx


def inverse_transform(x):
    return einsum("im,mjl,...ml->...ij", g.f, g.p, x)


def dx(u):
    n = 2 * g.m - 1
    y = u[:, 1:n, :]
    z = u[:, :n - 1, :]
    u_do = jax.lax.pad(y, 0.0, ((0, 0, 0), (0, 1, 0), (0, 0, 0)))
    u_up = jax.lax.pad(z, 0.0, ((0, 0, 0), (1, 0, 0), (0, 0, 0)))
    i = np.c_[:n]
    return (i + 1) // 2 * jnp.where(i % 2, u_do, -u_up)


def dy(x):
    l = g.l0[None, :]
    zm = (l - 1) * g.a * x
    zp = -(l + 2) * g.b * x
    return pad(zm, zp)


def dy_cos(x):
    l = g.l0[None, :]
    zm = (l + 1) * g.a * x
    zp = -l * g.b * x
    return pad(zm, zp)


def pad(zm, zp):
    return jax.lax.pad(zm[:, :, 1:g.l], 0.0,
                       ((0, 0, 0), (0, 0, 0),
                        (0, 1, 0))) + jax.lax.pad(zp[:, :, :g.l - 1], 0.0,
                                                  ((0, 0, 0), (0, 0, 0),
                                                   (1, 0, 0)))


def runge_kutta(y):
    a_ex = [1 / 3], [1 / 6, 1 / 2], [1 / 2, -1 / 2, 1]
    a_im = [1 / 6, 1 / 6], [1 / 3, 0, 1 / 3], [3 / 8, 0, 3 / 8, 1 / 4]
    b_ex = 1 / 2, -1 / 2, 1, 0
    b_im = 3 / 8, 0, 3 / 8, 1 / 4
    n = len(b_ex)
    f = [None] * n
    h = [None] * n
    f[0] = F(y)
    h[0] = G(y)
    for i in range(1, n):
        ex = g.dt * sum(a_ex[i - 1][j] * f[j]
                        for j in range(i) if a_ex[i - 1][j])
        im = g.dt * sum(a_im[i - 1][j] * h[j]
                        for j in range(i) if a_im[i - 1][j])
        Y = G_inv(y + ex + im, g.dt * a_im[i - 1][i])
        if any(a_ex[j][i] for j in range(i, n - 1)) or b_ex[i]: f[i] = F(Y)
        if any(a_im[j][i] for j in range(i, n - 1)) or b_im[i]: h[i] = G(Y)
    ex = g.dt * sum(b_ex[j] * f[j] for j in range(n) if b_ex[j])
    im = g.dt * sum(b_im[j] * h[j] for j in range(n) if b_im[j])
    return y + ex + im


def F(s):

    def hadv(x):
        return -dx(transform(u * x * g.sec2)) - dy(transform(v * x * g.sec2))

    def vadv(w, x):
        wt = np.zeros((1, g.nx, g.ny))
        dx = x[1:] - x[:-1]
        xd = dx * (1 / g.dz)[:, None, None]
        wx = jnp.r_[wt, xd * w, wt]
        return -0.5 * (wx[1:] + wx[:-1])

    def omega(x):
        f = jax.lax.cumsum(x * g.thick[:, None, None])
        alpha = g.alpha[:, None, None]
        pad = (1, 0), (0, 0), (0, 0)
        return (alpha * f +
                jnp.pad(alpha * f, pad)[:-1, ...]) / g.thick[:, None, None]

    vo = inverse_transform(s[g.vo])
    di = inverse_transform(s[g.di])
    te = inverse_transform(s[g.te])
    hu = inverse_transform(s[g.hu])
    wo = inverse_transform(s[g.wo])
    ic = inverse_transform(s[g.ic])

    psi = s[g.vo] * g.inv_eig
    chi = s[g.di] * g.inv_eig

    dchi_dx = dx(chi)
    dchi_dy_cos = dy_cos(chi)
    dpsi_dx = dx(psi)
    dpsi_dy_cos = dy_cos(psi)

    u = inverse_transform(dchi_dx - dpsi_dy_cos)
    v = inverse_transform(dchi_dy_cos + dpsi_dx)

    sp_dx = inverse_transform(dx(s[g.sp]))
    sp_dy = inverse_transform(dy_cos(s[g.sp]))

    u_dot_grad_sp = u * sp_dx * g.sec2 + v * sp_dy * g.sec2

    int_div = jax.lax.cumsum((di + u_dot_grad_sp) * g.thick[:, None, None])
    sigma = np.cumsum(g.thick)[:, None, None]
    dot_sigma = (sigma * int_div[-1] - int_div)[:-1]

    coriolis = np.tile(g.sin_y, (g.nx, 1))
    abs_vo = vo + coriolis

    vort_u = -v * abs_vo * g.sec2
    vort_v = u * abs_vo * g.sec2

    vadv_u = -vadv(dot_sigma, u)
    vadv_v = -vadv(dot_sigma, v)

    RT = g.r_gas * te
    sp_force_u = RT * sp_dx
    sp_force_v = RT * sp_dy

    force_u = vort_u + (vadv_u + sp_force_u) * g.sec2
    force_v = vort_v + (vadv_v + sp_force_v) * g.sec2

    force_u_spec = transform(force_u)
    force_v_spec = transform(force_v)

    dvo = -dx(force_v_spec) + dy(force_u_spec)
    ddi = -dx(force_u_spec) - dy(force_v_spec)

    ke = 0.5 * g.sec2 * (u**2 + v**2)
    dke = g.eig * transform(ke)
    doro = gravity_acceleration * (g.eig * g.orography)
    ddi += dke + doro

    dte_hadv = hadv(te)
    dte_vadv = vadv(dot_sigma, te)

    omega_mean = omega(u_dot_grad_sp)
    omega_full = omega(di + u_dot_grad_sp)
    dte_adiab = kappa * (g.temp[..., None, None] *
                         (u_dot_grad_sp - omega_mean) + te *
                         (u_dot_grad_sp - omega_full))
    dte = transform(te * di + dte_vadv + dte_adiab) + dte_hadv

    dhu_hadv = hadv(hu)
    dwo_hadv = hadv(wo)
    dic_hadv = hadv(ic)

    dhu_vadv = vadv(dot_sigma, hu)
    dwo_vadv = vadv(dot_sigma, wo)
    dic_vadv = vadv(dot_sigma, ic)

    dhu_dil = hu * di
    dwo_dil = wo * di
    dic_dil = ic * di

    dmoist_vadv = jnp.r_[dhu_vadv, dwo_vadv, dic_vadv]
    dmoist_dil = jnp.r_[dhu_dil, dwo_dil, dic_dil]
    dmoist_hadv = jnp.r_[dhu_hadv, dwo_hadv, dic_hadv]
    dmoist = transform(dmoist_vadv + dmoist_dil) + dmoist_hadv

    dsp_phys = -jnp.sum(
        g.thick[:, None, None] * u_dot_grad_sp, axis=0, keepdims=True)
    dsp = transform(dsp_phys)

    mask = np.r_[[1] * (g.l - 1), 0]

    return jnp.r_[dvo * mask, ddi * mask, dte * mask, dsp * mask,
                  dmoist * mask]


def G(s):
    shape = g.nz, 2 * g.m - 1, g.l
    tscale = 3 * g.nz, 2 * g.m - 1, g.l
    ddi = g.eig * (einsum("gh,hml->gml", g.geo, s[g.te]) +
                   g.r_gas * g.temp[..., None, None] * s[g.sp])
    dtesp = einsum("gh,hml->gml", jnp.r_[-g.tew, -g.thick[None]], s[g.di])
    return jnp.r_[jnp.zeros(shape), ddi, dtesp, jnp.zeros(tscale)]


def G_inv(s, dt):
    I = np.r_[[np.eye(g.nz)] * g.l]
    A = -dt * g.eig[:, None, None] * g.geo[None]
    B = -dt * g.r_gas * g.eig[:, None, None] * g.temp[None, :, None]
    C = dt * np.r_[[g.tew] * g.l]
    D = dt * np.c_[[[g.thick]] * g.l]
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


GRAVITY_ACCELERATION = 9.80616
uL = 6.37122e6
uT = 1 / 2 / 7.292e-5
einsum = functools.partial(jnp.einsum, precision=jax.lax.Precision.HIGHEST)
gravity_acceleration = GRAVITY_ACCELERATION / (uL / uT**2)
kappa = 2 / 7
g.r_gas = kappa * 0.0011628807950492582
g.m = 171
g.l = 172
g.nx = 512
g.ny = 256
g.nz = 32
g.zb = np.linspace(0, 1, g.nz + 1)
g.zc = (g.zb[1:] + g.zb[:-1]) / 2
g.thick = np.diff(g.zb)
g.dz = np.diff(g.zc)
dft = scipy.linalg.dft(g.nx)[:, :g.m] / np.sqrt(np.pi)
g.f = np.empty((g.nx, 2 * g.m - 1))
g.f[:, 0] = 1 / np.sqrt(2 * np.pi)
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
    M = min(g.m, g.l - k)
    m = np.c_[:M]
    m2 = m**2
    mk2 = (m + k)**2
    mkp2 = (m + k - 1)**2
    a = np.sqrt((4 * mk2 - 1) / (mk2 - m2))
    b = np.sqrt((mkp2 - m2) / (4 * mkp2 - 1))
    y[k, :M] = a * (g.sin_y * y[k - 1, :M] - b * y[k - 2, :M])
r = np.transpose(y, (1, 2, 0))
p = np.zeros((g.m, g.ny, g.l))
for m in range(g.m):
    p[m, :, m:g.l] = r[m, :, 0:g.l - m]
p = np.repeat(p, 2, axis=0)
g.p = p[1:]
g.w = 2 * np.pi * w / g.nx
g.temp = np.full((g.nz, ), 250)

p = np.r_[1:g.m]
q = np.c_[p, -p]
m, l = np.meshgrid(np.r_[0, q.ravel()], np.r_[:g.l], indexing="ij")
mask = abs(m) <= l
g.a = np.sqrt(mask * (l**2 - m**2) / (4 * l**2 - 1))
g.a[:, 0] = 0
g.b = np.sqrt(mask * ((l + 1)**2 - m**2) / (4 * (l + 1)**2 - 1))
g.b[:, -1] = 0
g.alpha = np.diff(np.log(g.zc), append=0) / 2
g.alpha[-1] = -np.log(g.zc[-1])

weights = np.zeros([g.nz, g.nz])
for j in range(g.nz):
    weights[j, j] = g.alpha[j]
    for k in range(j + 1, g.nz):
        weights[j, k] = g.alpha[k] + g.alpha[k - 1]
g.geo = g.r_gas * weights
p = np.tril(np.ones([g.nz, g.nz]))
alpha = g.alpha[..., None]
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
g.tew = (h0 - k - k_shifted) * g.thick
g.l0 = np.r_[:g.l]
g.eig = g.l0 * (g.l0 + 1)
g.inv_eig = np.r_[0, -1 / g.eig[1:]]

output_level_indices = [g.nz // 4, g.nz // 2, 3 * g.nz // 4, -1]
y_deg = np.rad2deg(np.arcsin(g.sin_y))
x_deg = np.linspace(0, 360, g.nx, endpoint=False)
a_zb, b_zb = np.loadtxt("ecmwf137_hybrid_levels.csv",
                        skiprows=1,
                        usecols=(1, 2),
                        delimiter="\t").T
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
if os.path.exists("s.raw") and os.path.exists("oro.raw"):
    s = np.fromfile("s.raw").reshape(shape)
    g.orography = np.fromfile("oro.raw", dtype=np.float32).reshape(shape[1:])
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
    nhyb = len(era["hybrid"].data)
    lat = era["latitude"].data
    lon = era["longitude"].data
    xi = np.meshgrid(y_deg, x_deg)
    xy_src = lat, lon
    sp = scipy.interpolate.interpn(xy_src, era["surface_pressure"].data, xi)
    oro = scipy.interpolate.interpn(xy_src,
                                    era["geopotential_at_surface"].data, xi)
    sp_nodal = sp[None, ...] / (1 / uL / uT**2)
    orography_input = oro[None, ...] / (uL * GRAVITY_ACCELERATION)
    M = {}
    for key, scale in [
        ("u_component_of_wind", uL / uT),
        ("v_component_of_wind", uL / uT),
        ("temperature", 1),
        ("specific_cloud_liquid_water_content", 1),
        ("specific_cloud_ice_water_content", 1),
        ("specific_humidity", 1),
    ]:
        val = np.empty((nhyb, g.nx, g.ny))
        for i in range(nhyb):
            val[i] = scipy.interpolate.interpn(xy_src, era[key].data[i], xi)
        source = a_zb[:, None, None] / sp + b_zb[:, None, None]
        upper = np.minimum(g.zb[1:, None, None, None], source[None, 1:, :, :])
        lower = np.maximum(g.zb[:-1, None, None, None],
                           source[None, :-1, :, :])
        weights = np.maximum(upper - lower, 0)
        weights /= np.sum(weights, axis=1, keepdims=True)
        M[key] = np.einsum("lnxy,nxy->lxy", weights, val / scale)
    cos = np.sqrt(1 - g.sin_y**2)
    u = transform(M["u_component_of_wind"] / cos)
    v = transform(M["v_component_of_wind"] / cos)
    s = np.empty(shape)
    vor = dx(v) - dy(u)
    div = dx(u) + dy(v)
    mask = np.r_[[1] * (g.l - 1), 0]
    s[g.vo] = vor * mask
    s[g.di] = div * mask
    s[g.te] = transform(M["temperature"] - g.temp.reshape(-1, 1, 1))
    s[g.sp] = transform(jnp.array(np.log(sp_nodal)))
    s[g.hu] = transform(M["specific_humidity"])
    s[g.wo] = transform(M["specific_cloud_liquid_water_content"])
    s[g.ic] = transform(M["specific_cloud_ice_water_content"])
    k = np.r_[:g.l] / (g.l - 1)
    g.orography = transform(jnp.array(orography_input)) * jnp.exp(-16 * k**4)
    s.tofile("s.raw")
    np.asarray(g.orography).tofile("oro.raw")

g.dt = 4.3752000000000006e-02
tau = 12900 / np.log2(g.ny / 128) / uT
scale = jnp.exp(-g.dt * g.eig**2 / (tau * g.eig[-1]**2))
out, *rest = jax.lax.scan(lambda x, _: (scale * runge_kutta(x), None),
                          s,
                          xs=None,
                          length=579)
np.asarray(out).tofile("out.raw")
