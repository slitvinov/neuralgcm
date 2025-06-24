import math
import matplotlib.pyplot as plt

a_ex = [[1 / 3], #
        [1 / 6, 1 / 2],
        [1 / 2, -1 / 2, 1]]
b_ex = [1 / 2, -1 / 2, 1, 0]

a_im = [[1 / 6, 1 / 6],
        [1 / 3, 0, 1 / 3],
        [3 / 8, 0, 3 / 8, 1 / 4]]
b_im = [3 / 8, 0, 3 / 8, 1 / 4]

num_steps = len(b_ex)

trace = []
C = 1
F = lambda t, y: math.sin(t)
G = lambda t, y: -C * y
# G_inv solves (I - gamma * G') Y = Y_star, where G' = diff(G, y)
G_inv = lambda Y_star, gamma: Y_star / (1 + C * gamma)
dt = 0.1
y0 = 1.0
nsteps = 50
istep = 0

while istep < nsteps:
    trace.append(y0)
    f = [None] * num_steps
    g = [None] * num_steps
    t = dt * istep
    istep += 1
    f[0] = F(t, y0)
    g[0] = G(t, y0)
    for i in range(1, num_steps):
        tau = sum(a_ex[i - 1])
        ex_terms = dt * sum(a_ex[i - 1][j] * f[j]
                            for j in range(i) if a_ex[i - 1][j])
        im_terms = dt * sum(a_im[i - 1][j] * g[j]
                            for j in range(i) if a_im[i - 1][j])
        Y_star = y0 + ex_terms + im_terms
        Y = G_inv(Y_star, dt * a_im[i - 1][i])
        if any(a_ex[j][i] for j in range(i, num_steps - 1)) or b_ex[i]:
            f[i] = F(t + tau * dt, Y)
        if any(a_im[j][i] for j in range(i, num_steps - 1)) or b_im[i]:
            g[i] = G(t + tau * dt, Y)
    ex_terms = dt * sum(b_ex[j] * f[j] for j in range(num_steps) if b_ex[j])
    im_terms = dt * sum(b_im[j] * g[j] for j in range(num_steps) if b_im[j])
    y0 += ex_terms + im_terms


def f(t):
    ''' 'diff(y, t) = -C*y + sin(t), y = 1
    '''
    p = C**2
    q = math.exp(C * t)
    return (C * q * math.sin(t) - q * math.cos(t) + p + 2) / (p + 1) / q


times = [i * dt for i in range(nsteps)]
plt.plot(times, trace, 'o')
plt.plot(times, [f(t) for t in times])
plt.savefig(f"runge.{C:6.2e}.png")
