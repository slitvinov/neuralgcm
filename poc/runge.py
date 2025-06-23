import numpy as np

a_ex = [[1 / 3], [1 / 6, 1 / 2], [1 / 2, -1 / 2, 1]]
a_im = [[1 / 6, 1 / 6], [1 / 3, 0, 1 / 3], [3 / 8, 0, 3 / 8, 1 / 4]]
b_ex = [1 / 2, -1 / 2, 1, 0]
b_im = [3 / 8, 0, 3 / 8, 1 / 4]
num_steps = len(b_ex)

trace = []
F = lambda x: x
G = lambda x: x
G_inv = lambda x, dt: x
dt = 0.1
y0 = 1
nsteps = 10

while len(trace) != nsteps:
    trace.append(y0)
    f = [None] * num_steps
    g = [None] * num_steps
    f[0] = F(y0)
    g[0] = G(y0)
    for i in range(1, num_steps):
        ex_terms = dt * sum(a_ex[i - 1][j] * f[j]
                            for j in range(i) if a_ex[i - 1][j])
        im_terms = dt * sum(a_im[i - 1][j] * g[j]
                            for j in range(i) if a_im[i - 1][j])
        Y_star = y0 + ex_terms + im_terms
        Y = G_inv(Y_star, dt * a_im[i - 1][i])
        if any(a_ex[j][i] for j in range(i, num_steps - 1)) or b_ex[i]:
            f[i] = F(Y)
        if any(a_im[j][i] for j in range(i, num_steps - 1)) or b_im[i]:
            g[i] = G(Y)
    ex_terms = dt * sum(b_ex[j] * f[j] for j in range(num_steps) if b_ex[j])
    im_terms = dt * sum(b_im[j] * g[j] for j in range(num_steps) if b_im[j])
    y0 += ex_terms + im_terms
    print(f)
    print(g)

print(trace, len(trace))
