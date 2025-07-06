import numpy as np
import scipy
import math
import jax.numpy as jnp
import jax
import functools

class g:
    pass

#@jax.jit
def modal(x):
    wx = g.w * x
    fwx = einsum("im,...ij->...mj", g.f, wx)
    return einsum("mjl,...mj->...ml", g.p, fwx)

#@jax.jit
def modal0(x):
    s0 = 1 / math.sqrt(2 * math.pi)
    s1 = 1 / math.sqrt(math.pi)
    F = np.fft.rfft(g.w * x, axis=1)
    a = np.empty((g.nz, 2 * g.m - 1, g.ny), dtype="float64")
    a[..., 0, :] = s0 * F[:, 0, :].real 
    a[:, 1::2, :] = s1 * F[:, 1:g.m, :].real
    a[:, 2::2, :] = -s1 * F[:, 1:g.m, :].imag
    return einsum("mjl,...mj->...ml", g.p, a)

#@jax.jit
def nodal(x):
    return einsum("im,mjl,...ml->...ij", g.f, g.p, x)

#@jax.jit
def nodal0(x):
    s0 = 1 / math.sqrt(2 * math.pi)
    s1 = 1 / math.sqrt(math.pi)
    s2 = g.nx / (2 * math.pi)
    u = einsum("mjl,...ml->...mj", g.p, x)
    F = np.empty((g.nz, g.nx // 2 + 1, g.ny), dtype="complex128")
    F[:, 0, :] = u[:, 0, :] / s0
    F[:, 1:g.m, :] = (u[:, 1::2, :] - 1j * u[:, 2::2, :]) / s1
    F[:, g.m:, :] = 0
    return np.fft.irfft(F, axis=1) * s2

#einsum = jnp.einsum
einsum = np.einsum
g.m = 171 // 2
g.nx = 512 // 2
g.l = g.m + 1
g.ny = 2 * g.nx
g.nz = 2

g.zb = np.linspace(0, 1, g.nz + 1)
g.zc = (g.zb[1:] + g.zb[:-1]) / 2
g.thick = np.diff(g.zb)
g.dz = np.diff(g.zc)
dft = scipy.linalg.dft(g.nx)[:, :g.m] / math.sqrt(math.pi)
g.f = np.empty((g.nx, 2 * g.m - 1))
g.f[:, 0] = 1 / math.sqrt(2 * math.pi)
g.f[:, 1::2] = np.real(dft[:, 1:])
g.f[:, 2::2] = -np.imag(dft[:, 1:])
g.sin_y, w = scipy.special.roots_legendre(g.ny)
g.sec2 = 1 / (1 - g.sin_y**2)
g.cos = np.sqrt(1 - g.sin_y**2)
y = np.zeros((g.l, g.m, g.ny))
y[0, 0] = 1 / np.sqrt(2)
for m in range(1, g.m):
    y[0, m] = -np.sqrt(1 + 1 / (2 * m)) * g.cos * y[0, m - 1]
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

n = np.random.rand(g.nz, g.nx, g.ny)
m0 = modal(n)
m1 = modal0(n)
u = np.unique(m0 / m1)
print("%.16e %.16e" % (np.min(u), np.max(u[1])))
#assert np.allclose(m0, m1, atol=0, rtol=1e-2)

n0 = nodal(m1)
n1 = nodal0(m1)
u = np.unique(n0 / n1)
print("%.16e %.16e" % (np.min(u), np.max(u[1])))
#assert np.allclose(n0, n1, atol=0, rtol=1e-5)
