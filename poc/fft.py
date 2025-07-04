import numpy as np
import scipy
import math
from numpy import einsum

class g:
    pass

def modal(x):
    wx = g.w * x
    fwx = einsum("im,...ij->...mj", g.f, wx)
    return fwx

def modal0(x):
    wx = g.w * x
    fwx = einsum("im,...ij->...mj", g.f, wx)
    return fwx

g.m = 171 // 4
g.nx = 512 // 4
g.l = g.m + 1
g.ny = 2 * g.nx
g.nz = 1

dft = scipy.linalg.dft(g.nx)[:, :g.m] / math.sqrt(math.pi)
g.f = np.empty((g.nx, 2 * g.m - 1))
g.f[:, 0] = 1 / math.sqrt(2 * math.pi)
g.f[:, 1::2] = np.real(dft[:, 1:])
g.f[:, 2::2] = -np.imag(dft[:, 1:])

g.sin_y, w = scipy.special.roots_legendre(g.ny)
g.w = 2 * math.pi * w / g.nx

n = np.random.rand(g.nz, g.nx, g.ny)
p = modal(n)
