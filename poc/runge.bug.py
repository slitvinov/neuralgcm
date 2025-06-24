import math
import statistics

def fun(t):
    return 2 * math.exp(t) - 1

a, b, c = 3/8, 3/8, 1/4
# a, b, c = -1/4, 3/4, 1/2

a_im = [[1 / 6, 1 / 6], [1 / 3, 0, 1 / 3], [a, 0, b, c]]
b_im = [a, 0, b, c]
num_steps = 4
G = lambda y: 1 + y
G_inv = lambda Y_star, gamma: Y_star / (1 + gamma)

for nsteps in (2 << x for x in range(3, 15)):
    trace = []
    y0 = 1.0
    dt = 1.0 / nsteps
    istep = 0
    while istep < nsteps:
        trace.append(y0)
        istep += 1
        g = [None] * num_steps
        g[0] = G(y0)
        for i in range(1, num_steps):
            im_terms = dt * sum(a_im[i - 1][j] * g[j] for j in range(i))
            Y_star = y0 + im_terms
            Y = G_inv(Y_star, dt * a_im[i - 1][i])
            g[i] = G(Y)
        im_terms = dt * sum(b_im[j] * g[j] for j in range(num_steps))
        y0 += im_terms
    times = [i * dt for i in range(nsteps)]
    diff = statistics.fmean((a - b)**2 for a, b in zip(trace, (fun(t) for t in times)))
    print(f"{nsteps:08d} {diff:+.16e}")
