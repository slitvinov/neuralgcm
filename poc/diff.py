import numpy as np
import sys

a = np.memmap(sys.argv[1], dtype=float32)
b = np.memmap(sys.argv[2], dtype=float32)

print(np.mean((a - b)**2))
