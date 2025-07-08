import matplotlib.pyplot as plt
import numpy as np

class g:
    pass

g.nx = 512
g.ny = 256
oro = np.fromfile("oro.raw").reshape((g.nx, g.ny))
plt.imsave(f"oro.png", oro.T, cmap="jet")
