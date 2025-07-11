import cartopy.crs as ccrs
import cartopy.feature as cfeature
import functools
import math
import matplotlib.pyplot as plt
import numpy as np
import re
import scipy
import sys
import os


def nodal(x):
    nz, *rest = np.shape(x)
    s0 = math.sqrt(math.pi) * math.sqrt(2)
    s1 = math.sqrt(math.pi)
    s2 = g.nx / s0**2
    u = np.einsum("mjl,...ml->...mj", g.p, x)
    F0 = u[:, :1, :] * s0
    F1 = (u[:, 1::2, :] - 1j * u[:, 2::2, :]) * s1
    Fpad = np.zeros((nz, g.nx // 2 + 1 - g.m, g.ny))
    F = np.r_['1', F0, F1, Fpad]
    return np.fft.irfft(F, axis=1) * s2


class g:
    pass


dtype = np.dtype("float32")
g.nz = 32
flt = dtype.itemsize
sz = os.path.getsize(sys.argv[1])
a = 12 * flt * g.nz + 2 * flt
b = 6 * flt * g.nz + flt
c = -sz - 6 * flt * g.nz - flt
D = b**2 - 4 * a * c
g.m = (-b + math.sqrt(D)) / (2 * a)
g.m = round(g.m)

g.l = g.m + 1
g.nx = 3 * g.m + 1
g.ny = g.nx // 2
g.f = np.empty((g.nx, 2 * g.m - 1))
dft = scipy.linalg.dft(g.nx)[:, :g.m] / math.sqrt(math.pi)
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

fig = plt.figure(figsize=(4, 2), dpi=300)
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
ax.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=0.3)
ax.set_xticks([])
ax.set_yticks([])
dummy_data = np.empty((g.nx, g.ny))
im = ax.imshow(dummy_data,
               extent=[0, 360, -90, 90],
               origin='upper',
               transform=ccrs.PlateCarree())
cbar = fig.colorbar(im, ax=ax, orientation="horizontal", shrink=0.8, pad=0.02)
cbar.ax.tick_params(labelsize=6, length=2)

for path in sys.argv[1:]:
    base = re.sub("[.]raw$", "", path)
    base = re.sub("^out[.]", "", base)
    print(shape)
    s = np.memmap(path, dtype=dtype).reshape(shape)
    for name, sli, diverging in (
        ("vo", g.vo, True),
        ("di", g.di, False),
        ("te", g.te, True),
        ("sp", g.sp, False),
        ("hu", g.hu, False),
        ("wo", g.wo, False),
        ("ic", g.ic, False),
    ):
        image = name + "." + base + ".png"
        sys.stderr.write(f"vis.py: {image}\n")
        fi = s[sli]
        nz, *rest = np.shape(fi)
        fi = nodal(fi[nz // 2][None])
        vmin = np.min(fi)
        vmax = np.max(fi)
        print(name, vmin, vmax, np.any(np.isnan(fi)))
        if diverging:
            vmax = max(abs(vmin), abs(vmax))
            vmin = -vmax
            cmap = "Spectral_r"
        else:
            cmap = "jet"
        im.set_data(fi.T)
        im.set_cmap(cmap)
        im.set_clim(vmin, vmax)
        im.norm = None
        cbar.update_normal(im)
        cbar.set_ticks([vmin, vmax])
        cbar.set_ticklabels([f"{vmin: 6.1}", f"{vmax: 6.1e}"])
        fig.savefig(image, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
