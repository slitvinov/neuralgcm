import matplotlib.pyplot as plt
import numpy as np
import sys

def nodal(x):
    return einsum("im,mjl,...ml->...ij", g.f, g.p, x)

s = np.fromfile(sys.argv[1], dtype=np.float32).reshape(shape)

hu = nodal(s[g.hu])[g.nz//2]
vmin = np.min(hu)
vmax = np.max(hu)
plt.imsave(f"oro.png", hu.T, cmap="jet", vmin=vmin, vmax=vmax)
