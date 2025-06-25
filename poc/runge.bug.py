import math
import statistics


def fun(t):
    return (A * math.exp(B * t) - A) / B


a, b, c = -1 / 4, 3 / 4, 1 / 2
#a, b, c = 3 / 8, 3 / 8, 1 / 4

A = 1
B = 2
a_im = [[1 / 6, 1 / 6], [1 / 3, 0, 1 / 3], [a, 0, b, c]]
b_im = [a, 0, b, c]
num_steps = 4
G = lambda y: A + B * y
G_inv = lambda Y_star, gamma: -Y_star / (-B * gamma - 1)

for nsteps in (2 << x for x in range(3, 12)):
    trace = []
    y0 = 0.0
    dt = 1.0 / nsteps
    istep = 0
    while istep < nsteps:
        trace.append(y0)
        istep += 1
        g = [None] * num_steps
        g[0] = G(y0)
        for i in range(1, num_steps):
            Y_star = y0 + dt * sum(a_im[i - 1][j] * g[j] for j in range(i))
            Y = G_inv(Y_star, dt * a_im[i - 1][i])
            g[i] = G(Y)
        y0 += dt * sum(b_im[j] * g[j] for j in range(num_steps))
    times = [i * dt for i in range(nsteps)]
    diff = statistics.fmean(
        (a - b)**2 for a, b in zip(trace, (fun(t) for t in times)))
    print(f"{nsteps:08d} {diff:.16e}")
