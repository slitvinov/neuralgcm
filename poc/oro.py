import matplotlib.pyplot as plt
import numpy as np

class g:
    pass

g.m = 171
g.l = g.m + 1
g.nx = 3 * g.m + 1
g.ny = g.nx // 2
oro = np.memmap("oro.raw", dtype=float).reshape((g.nx, g.ny))
oro = np.flipud(oro.T)
# oro = np.roll(oro, -g.nx//5, axis=1)
# oro = np.roll(oro, g.ny//4, axis=0)
plt.imsave(f"oro.png", oro, cmap="jet")
