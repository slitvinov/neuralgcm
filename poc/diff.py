import numpy as np
import sys

a = np.memmap(sys.argv[1], dtype=np.float32)
b = np.memmap(sys.argv[2], dtype=np.float32)

print(np.mean((a - b)**2))
