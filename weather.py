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


def vadvection(w, x):
    wt = np.zeros((1, g.longitude_nodes, g.latitude_nodes))
    dx = x[1:] - x[:-1]
    xd = dx * (1 / g.center_to_center)[:, None, None]
    wx = jnp.r_[wt, xd * w, wt]
    return -0.5 * (wx[1:] + wx[:-1])


def real_basis_derivative(u):
    n = 2 * g.longitude_wavenumbers - 1
    y = u[:, 1:n, :]
    z = u[:, :n - 1, :]
    u_do = jax.lax.pad(y, 0.0, ((0, 0, 0), (0, 1, 0), (0, 0, 0)))
    u_up = jax.lax.pad(z, 0.0, ((0, 0, 0), (1, 0, 0), (0, 0, 0)))
    i = np.c_[:n]
    return (i + 1) // 2 * jnp.where(i % 2, u_do, -u_up)


def cos_lat_d_dlat(x):
    l0 = np.arange(g.total_wavenumbers)
    l = np.tile(l0, (2 * g.longitude_wavenumbers - 1, 1))
    zm = (l + 1) * g.a * x
    zp = -l * g.b * x
    lm1 = jax.lax.pad(zm[:, :, 1:g.total_wavenumbers], 0.0,
                      ((0, 0, 0), (0, 0, 0), (0, 1, 0)))
    lp1 = jax.lax.pad(zp[:, :, :g.total_wavenumbers - 1], 0.0,
                      ((0, 0, 0), (0, 0, 0), (1, 0, 0)))
    return lm1 + lp1


def sec_lat_d_dlat_cos2(x):
    l0 = np.arange(g.total_wavenumbers)
    l = np.tile(l0, (2 * g.longitude_wavenumbers - 1, 1))
    zm = (l - 1) * g.a * x
    zp = -(l + 2) * g.b * x
    lm1 = jax.lax.pad(zm[:, :, 1:g.total_wavenumbers], 0.0,
                      ((0, 0, 0), (0, 0, 0), (0, 1, 0)))
    lp1 = jax.lax.pad(zp[:, :, :g.total_wavenumbers - 1], 0.0,
                      ((0, 0, 0), (0, 0, 0), (1, 0, 0)))
    return lm1 + lp1


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


def omega(g_term):
    f = jax.lax.cumsum(g_term * g.thick[:, None, None])
    alpha = g.alpha[:, None, None]
    pad = (1, 0), (0, 0), (0, 0)
    return (alpha * f + jnp.pad(alpha * f, pad)[:-1, ...]) / g.thick[:, None,
                                                                     None]


def F(s):

    def hadvection(s):
        sec2 = 1 / (1 - g.sin_lat**2)
        m = transform(u * s * sec2)
        n = transform(v * s * sec2)
        return -real_basis_derivative(m) - sec_lat_d_dlat_cos2(n)

    vort = inverse_transform(s[g.vo])
    div = inverse_transform(s[g.di])
    temp = inverse_transform(s[g.te])
    hu = inverse_transform(s[g.hu])
    wa = inverse_transform(s[g.wo])
    ic = inverse_transform(s[g.ic])
    l = np.arange(1, g.total_wavenumbers)
    inverse_eigenvalues = np.zeros(g.total_wavenumbers)
    inverse_eigenvalues[1:] = -1 / (l * (l + 1))
    stream_function = s[g.vo] * inverse_eigenvalues
    velocity_potential = s[g.di] * inverse_eigenvalues

    c00 = real_basis_derivative(velocity_potential)
    c01 = cos_lat_d_dlat(velocity_potential)
    c10 = real_basis_derivative(stream_function)
    c11 = cos_lat_d_dlat(stream_function)
    v0 = c00 - c11
    v1 = c01 + c10
    u = inverse_transform(v0)
    v = inverse_transform(v1)
    grad_u = inverse_transform(real_basis_derivative(s[g.sp]))
    grad_v = inverse_transform(cos_lat_d_dlat(s[g.sp]))
    sec2 = 1 / (1 - g.sin_lat**2)
    u_dot_grad = u * grad_u * sec2 + v * grad_v * sec2
    f_exp = jax.lax.cumsum(u_dot_grad * g.thick[:, None, None])
    f_full = jax.lax.cumsum((div + u_dot_grad) * g.thick[:, None, None])
    sum_sigma = np.cumsum(g.thick)[:, None, None]
    sigma_exp = (sum_sigma * f_exp[-1] - f_exp)[:-1]
    sigma_full = (sum_sigma * f_full[-1] - f_full)[:-1]
    coriolis = np.tile(g.sin_lat, (g.longitude_nodes, 1))
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

    t1 = hadvection(temp)
    hu_h1 = hadvection(hu)
    wa_h1 = hadvection(wa)
    ic_h1 = hadvection(ic)
    t0 = temp * div

    temp_vert = vadvection(sigma_full, temp)
    # np.unique(g.temp[..., None, None].ravel()).size > 1:

    t_mean = g.temp[..., None, None] * (u_dot_grad - omega(u_dot_grad))
    t_var = temp * (u_dot_grad - omega(div + u_dot_grad))
    temp_adiab = kappa * (t_mean + t_var)

    xds = g.thick[:, None, None] * u_dot_grad
    logsp_tendency = -xds.sum(axis=0, keepdims=True)

    hu_v = vadvection(sigma_full, hu)
    wa_v = vadvection(sigma_full, wa)
    ic_v = vadvection(sigma_full, ic)
    v = jnp.r_[hu_v, wa_v, ic_v]
    h0 = jnp.r_[hu, wa, ic] * div
    h1 = jnp.r_[hu_h1, wa_h1, ic_h1]

    mask = np.r_[[1] * (g.total_wavenumbers - 1), 0]
    return jnp.r_[vort_tendency * mask,
                  (div_tendency + ke_tendency + oro_tendency) * mask,
                  (transform(t0 + temp_vert + temp_adiab) + t1) * mask,
                  transform(logsp_tendency) * mask,
                  (transform(v + h0) + h1) * mask]


def G(s):
    shape = g.layers, 2 * g.longitude_wavenumbers - 1, g.total_wavenumbers
    tscale = 3 * g.layers, 2 * g.longitude_wavenumbers - 1, g.total_wavenumbers
    l0 = np.arange(g.total_wavenumbers)
    di = l0 * (l0 + 1) * (einsum("gh,hml->gml", g.geo, s[g.te]) +
                          r_gas * g.temp[..., None, None] * s[g.sp])
    tesp = einsum("gh,hml->gml", jnp.r_[-g.tew, -g.thick[None]], s[g.di])
    return jnp.r_[jnp.zeros(shape), di, tesp, jnp.zeros(tscale)]


def G_inv(s, dt):
    l = g.total_wavenumbers
    j = g.layers
    l0 = np.r_[:l]
    lam = -l0 * (l0 + 1)
    I = np.r_[[np.eye(j)] * l]
    A = dt * lam[:, None, None] * g.geo[None]
    B = dt * r_gas * lam[:, None, None] * g.temp[None, :, None]
    C = dt * np.r_[[g.tew] * l]
    D = dt * np.c_[[[g.thick]] * l]
    Z = np.zeros([l, j, 1])
    Z0 = np.zeros([l, 1, j])
    I0 = np.ones([l, 1, 1])
    row0 = np.c_[I, A, B]
    row1 = np.c_[C, I, Z]
    row2 = np.c_[D, Z0, I0]
    inv = np.linalg.inv(np.r_['1', row0, row1, row2])
    M = einsum("lgh,hml->gml", inv, s[g.ditesp])
    return jnp.r_[s[g.vo], M, s[g.hu], s[g.wo], s[g.ic]]


def open(path):
    x = xarray.open_zarr(path, chunks=None, storage_options=dict(token="anon"))
    return x.sel(time="19900501T00")


GRAVITY_ACCELERATION = 9.80616
uL = 6.37122e6
uT = 1 / 2 / 7.292e-5
einsum = functools.partial(jnp.einsum, precision=jax.lax.Precision.HIGHEST)
gravity_acceleration = GRAVITY_ACCELERATION / (uL / uT**2)
kappa = 2 / 7
r_gas = kappa * 0.0011628807950492582
g.longitude_wavenumbers = 171
g.total_wavenumbers = 172
g.longitude_nodes = 512
g.latitude_nodes = 256
g.layers = 32
g.boundaries = np.linspace(0, 1, g.layers + 1)
g.centers = (g.boundaries[1:] + g.boundaries[:-1]) / 2
g.thick = np.diff(g.boundaries)
g.center_to_center = np.diff(g.centers)
dft = scipy.linalg.dft(
    g.longitude_nodes)[:, :g.longitude_wavenumbers] / np.sqrt(np.pi)
g.f = np.empty((g.longitude_nodes, 2 * g.longitude_wavenumbers - 1))
g.f[:, 0] = 1 / np.sqrt(2 * np.pi)
g.f[:, 1::2] = np.real(dft[:, 1:])
g.f[:, 2::2] = -np.imag(dft[:, 1:])
g.sin_lat, w = scipy.special.roots_legendre(g.latitude_nodes)
q = np.sqrt(1 - g.sin_lat * g.sin_lat)
y = np.zeros((g.total_wavenumbers, g.longitude_wavenumbers, g.latitude_nodes))
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
    y[k, :M] = a * (g.sin_lat * y[k - 1, :M] - b * y[k - 2, :M])
r = np.transpose(y, (1, 2, 0))
p = np.zeros((g.longitude_wavenumbers, g.latitude_nodes, g.total_wavenumbers))
for m in range(g.longitude_wavenumbers):
    p[m, :, m:g.total_wavenumbers] = r[m, :, 0:g.total_wavenumbers - m]
p = np.repeat(p, 2, axis=0)
g.p = p[1:]
g.w = 2 * np.pi * w / g.longitude_nodes
g.temp = np.full((g.layers, ), 250)

p = np.r_[1:g.longitude_wavenumbers]
q = np.c_[p, -p]
m, l = np.meshgrid(np.r_[0, q.ravel()],
                   np.r_[:g.total_wavenumbers],
                   indexing="ij")
mask = abs(m) <= l
g.a = np.sqrt(mask * (l**2 - m**2) / (4 * l**2 - 1))
g.a[:, 0] = 0
g.b = np.sqrt(mask * ((l + 1)**2 - m**2) / (4 * (l + 1)**2 - 1))
g.b[:, -1] = 0
g.alpha = np.diff(np.log(g.centers), append=0) / 2
g.alpha[-1] = -np.log(g.centers[-1])

weights = np.zeros([g.layers, g.layers])
for j in range(g.layers):
    weights[j, j] = g.alpha[j]
    for k in range(j + 1, g.layers):
        weights[j, k] = g.alpha[k] + g.alpha[k - 1]
g.geo = r_gas * weights
p = np.tril(np.ones([g.layers, g.layers]))
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

output_level_indices = [g.layers // 4, g.layers // 2, 3 * g.layers // 4, -1]
desired_lat = np.rad2deg(np.arcsin(g.sin_lat))
desired_lon = np.linspace(0, 360, g.longitude_nodes, endpoint=False)
a_boundaries, b_boundaries = np.loadtxt("ecmwf137_hybrid_levels.csv",
                                        skiprows=1,
                                        usecols=(1, 2),
                                        delimiter="\t").T
n = g.layers
g.vo = np.s_[:n]
g.di = np.s_[n:2 * n]
g.te = np.s_[2 * n:3 * n]
g.sp = np.s_[3 * n:3 * n + 1]
g.hu = np.s_[3 * n + 1:4 * n + 1]
g.wo = np.s_[4 * n + 1:5 * n + 1]
g.ic = np.s_[5 * n + 1:6 * n + 1]
g.ditesp = np.s_[n:3 * n + 1]
shape = 6 * g.layers + 1, 2 * g.longitude_wavenumbers - 1, g.total_wavenumbers
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
    xi = np.meshgrid(desired_lat, desired_lon)
    points = lat, lon
    sp = scipy.interpolate.interpn(points, era["surface_pressure"].data, xi)
    oro = scipy.interpolate.interpn(points,
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
        val = np.empty((nhyb, g.longitude_nodes, g.latitude_nodes))
        for i in range(nhyb):
            val[i] = scipy.interpolate.interpn(points, era[key].data[i], xi)
        source = a_boundaries[:, None, None] / sp + b_boundaries[:, None, None]
        upper = np.minimum(g.boundaries[1:, None, None, None],
                           source[None, 1:, :, :])
        lower = np.maximum(g.boundaries[:-1, None, None, None],
                           source[None, :-1, :, :])
        weights = np.maximum(upper - lower, 0)
        weights /= np.sum(weights, axis=1, keepdims=True)
        M[key] = np.einsum("lnxy,nxy->lxy", weights, val / scale)
    cos = np.sqrt(1 - g.sin_lat**2)
    u = transform(M["u_component_of_wind"] / cos)
    v = transform(M["v_component_of_wind"] / cos)
    s = np.empty(shape)
    vor = real_basis_derivative(v) - sec_lat_d_dlat_cos2(u)
    div = real_basis_derivative(u) + sec_lat_d_dlat_cos2(v)
    mask = np.r_[[1] * (g.total_wavenumbers - 1), 0]
    s[g.vo] = vor * mask
    s[g.di] = div * mask
    s[g.te] = transform(M["temperature"] - g.temp.reshape(-1, 1, 1))
    s[g.sp] = transform(jnp.array(np.log(sp_nodal)))
    s[g.hu] = transform(M["specific_humidity"])
    s[g.wo] = transform(M["specific_cloud_liquid_water_content"])
    s[g.ic] = transform(M["specific_cloud_ice_water_content"])
    k = np.r_[:g.total_wavenumbers] / (g.total_wavenumbers - 1)
    g.orography = transform(jnp.array(orography_input)) * jnp.exp(-16 * k**4)
    s.tofile("s.raw")
    np.asarray(g.orography).tofile("oro.raw")

g.dt = 4.3752000000000006e-02
tau = 12900 / np.log2(g.latitude_nodes / 128) / uT
l0 = np.r_[:g.total_wavenumbers]
eig = l0 * (l0 + 1)
scale = jnp.exp(-g.dt * eig**2 / (tau * eig[-1]**2))
out, *rest = jax.lax.scan(lambda x, _: (scale * runge_kutta(x), None),
                          s,
                          xs=None,
                          length=579)
np.asarray(out).tofile("out.raw")
