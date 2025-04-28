from dinosaur import jax_numpy_utils
import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg


def real_basis(wavenumbers, nodes):
    dft = scipy.linalg.dft(nodes)[:, :wavenumbers] / np.sqrt(np.pi)
    cos = np.real(dft[:, 1:])
    sin = -np.imag(dft[:, 1:])
    f = np.empty(shape=[nodes, 2 * wavenumbers - 1], dtype=np.float64)
    f[:, 0] = 1 / np.sqrt(2 * np.pi)
    f[:, 1::2] = cos
    f[:, 2::2] = sin
    return f


def real_basis_derivative(u, /, axis=-1):
    i = jnp.arange(u.shape[axis]).reshape((-1, ) + (1, ) * (-1 - axis))
    j = (i + 1) // 2
    u_down = jax_numpy_utils.shift(u, -1, axis)
    u_up = jax_numpy_utils.shift(u, +1, axis)
    return j * jnp.where(i % 2, u_down, -u_up)


def quadrature_nodes(nodes):
    xs = np.linspace(0, 2 * np.pi, nodes, endpoint=False)
    weights = 2 * np.pi / nodes
    return xs, weights
