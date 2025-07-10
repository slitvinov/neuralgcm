import numpy as np
import sys

a = np.memmap(sys.argv[1], dtype=np.float32)
print(np.min(a), np.max(a), np.mean(a))
