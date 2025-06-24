import math


def f(t):
    return math.exp(t) - 1


a_ex = [
    [1 / 3],  #
    [1 / 6, 1 / 2],
    [1 / 2, -1 / 2, 1]
]
b_ex = [1 / 2, -1 / 2, 1, 0]

a, b, c = -1 / 4, 3 / 4, 1 / 2
# a, b, c = 3 / 8, 3 / 8, 1 / 4
a_im = [[1 / 6, 1 / 6], [1 / 3, 0, 1 / 3], [a, 0, b, c]]
b_im = [a, 0, b, c]
num_steps = len(b_ex)

trace = []
G = lambda y: 1 + y
G_inv = lambda Y_star, gamma: -Y_star / (gamma - 1)

for nsteps in (2 << x for x in range(3, 12)):
    dt = 5.0 / nsteps
    istep = 0
    y0 = 0.0
    g = [None] * num_steps
    t = dt * istep
    istep += 1
    g[0] = G(y0)
    for i in range(1, num_steps):
        im_terms = dt * sum(a_im[i - 1][j] * g[j]
                            for j in range(i) if a_im[i - 1][j])
        Y_star = y0 + im_terms
        Y = G_inv(Y_star, dt * a_im[i - 1][i])
        if any(a_im[j][i] for j in range(i, num_steps - 1)) or b_im[i]:
            g[i] = G(Y)
    im_terms = dt * sum(b_im[j] * g[j] for j in range(num_steps) if b_im[j])
    y0 += im_terms
    diff = abs(y0 - f(dt)) / dt
    print(f"{nsteps:6d} {diff:.2e}")
