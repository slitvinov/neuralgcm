import cartopy.crs as ccrs
import cartopy.feature as cfeature
import functools
import jax
import jax.numpy as jnp
import math
import matplotlib.pyplot as plt
import numpy as np
import re
import scipy
import sys


@jax.jit
def nodal(x):
    return einsum("im,mjl,...ml->...ij", g.f, g.p, x)


class g:
    pass


einsum = functools.partial(jnp.einsum, precision=jax.lax.Precision.HIGHEST)
g.m = 171
g.l = g.m + 1
g.nx = 3 * g.m + 1
g.ny = g.nx // 2
g.nz = 32
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
dummy_data = np.zeros((g.nx, g.ny)).T
im = ax.imshow(dummy_data,
               extent=[0, 360, -90, 90],
               origin='upper',
               transform=ccrs.PlateCarree())
cbar = fig.colorbar(im, ax=ax, orientation="horizontal", shrink=0.8, pad=0.02)
cbar.ax.tick_params(labelsize=6, length=2)

for path in sys.argv[1:]:
    base = re.sub("[.]raw$", "", path)
    base = re.sub("^out[.]", "", base)
    s = np.fromfile(path, dtype=np.float32).reshape(shape)
    for name, sli, diverging in (
        ("vo", g.vo, True),
        ("di", g.di, False),
        ("te", g.te, True),
        ("hu", g.hu, False),
        ("wo", g.wo, False),
        ("ic", g.ic, False),
    ):
        image = name + "." + base + ".png"
        sys.stderr.write(f"vis.py: {image}\n")
        fi = s[sli][g.nz // 2]
        fi = nodal(fi)
        vmin = np.min(fi)
        vmax = np.max(fi)
        if diverging:
            vmax = max(abs(vmin), abs(vmax))
            vmin = -vmax
            cmap = "Spectral_r"
        else:
            cmap = "jet"
        im.set_data(fi)
        im.set_cmap(cmap)
        im.set_clim(vmin, vmax)
        cbar.update_normal(im)
        fig.savefig(image, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
