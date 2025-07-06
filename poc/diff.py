import numpy as np
import sys

a = np.memmap(sys.argv[1], dtype=np.float32)
b = np.memmap(sys.argv[2], dtype=np.float32)

diff = (a - b)**2)

print(np.min(diff), np.max(diff), np.mean(diff))
