import matplotlib.pyplot as plt
import numpy as np


oro = np.fromfile("oro.raw", dtype=np.float32).reshape(shape)
vmin = np.min(hu)
vmax = np.max(hu)
plt.imsave(f"oro.png", hu.T, cmap="jet", vmin=vmin, vmax=vmax)
