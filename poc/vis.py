import matplotlib.pyplot as plt
i = 0
while True:
    hu = nodal(s[g.hu])[g.nz//2]
    if i == 0:
        vmin = np.min(hu)
        vmax = np.max(hu)
    plt.imsave(f"hu.{i:08d}.png", hu.T, cmap="jet", vmin=vmin, vmax=vmax)
    i += 1
    if i == 100:
        break
    out, *rest = jax.lax.scan(lambda x, _: (scale * runge_kutta(x), None),
                              s,
                              xs=None,
                              length=100)
    s = out
