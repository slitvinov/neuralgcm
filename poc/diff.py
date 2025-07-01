import numpy as np
import sys

a = np.memmap(sys.argv[1])
b = np.memmap(sys.argv[2])

print(np.linalg.norm(a - b))
