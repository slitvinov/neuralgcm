import math
import matplotlib.pyplot as plt

a_ex = [[1 / 3], [1 / 6, 1 / 2], [1 / 2, -1 / 2, 1]]
a_im = [[1 / 6, 1 / 6], [1 / 3, 0, 1 / 3], [3 / 8, 0, 3 / 8, 1 / 4]]
b_ex = [1 / 2, -1 / 2, 1, 0]
b_im = [3 / 8, 0, 3 / 8, 1 / 4]
num_steps = len(b_ex)

trace = []
C = 0.1
F = lambda y: math.sin(dt * istep)
G = lambda y: - C * y
dt = 0.1
y0 = 1.0
nsteps = 50
istep = 0

while istep < nsteps:
    trace.append(y0)
    istep += 1
    y0 += dt * (F(y0) + G(y0))

def f(t):
    ''' 'diff(y, t) = -C*y + sin(t), y = 1
    '''
    p = C**2
    q = math.exp(C*t)
    return (C * q * math.sin(t) - q * math.cos(t) + p + 2) / (p + 1) / q

times = [i * dt for i in range(nsteps)]
plt.plot(times, trace, 'o')
plt.plot(times, [f(t) for t in times])
plt.savefig(f"euler.{C:6.2e}.png")
