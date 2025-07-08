import matplotlib.pyplot as plt
import numpy as np

class g:
    pass

g.nx = 512
g.ny = 256
oro = np.memmap("oro.raw", dtype=float).reshape((g.nx, g.ny))
oro = np.flipud(oro.T)
oro = np.roll(oro, -g.nx//5, axis=1)
# oro = np.roll(oro, g.ny//4, axis=0)
plt.imsave(f"oro.png", oro, cmap="jet")
