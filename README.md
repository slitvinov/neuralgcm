## Install

```
python -m venv ~/.venv/neuralgcm
. ~/.venv/neuralgcm/bin/activate
python -m pip install jax[cuda12] gcsfs matplotlib pint tree_math xarray zarr
```

## Test

Save reference raw files

```
for i in *.raw; do cmp $i ~/$i; echo $i $?; done
```

## References


- https://github.com/neuralgcm/dinosaur
- https://www.nature.com/articles/s41586-024-07744-y
- https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-07744-y/MediaObjects/41586_2024_7744_MOESM1_ESM.pdf
- Durran, D. R. Numerical methods for fluid dynamics: With
  applications to geophysics Second edn, Vol. 32 (Springer, New
  York,2010)

The dynamical core of NeuralGCM solves the primitive equations, which
represent a combination of (1) momentum equations, (2) the second law
of thermodynamics, (3) a thermodynamic equation of state (ideal gas),
(4) continuity equation and (5) hydrostatic approximation. For solving
the equations we use a divergence-vorticity representation of the
horizontal winds, resulting in equations for the following seven
prognostic variables: divergence $\delta$, vorticity $\zeta$,
temperature $T$, logarithm of the surface pressure $\log p_{s}$, as
well as $3$ moisture species (specific humidity $q$, specific cloud
ice $q_{c_i}$ and specific liquid cloud water content $q_{c_l}$). To
facilitate efficient time integration of our models we split
temperature $T$ into a uniform reference temperature on each sigma
level $\bar{T}_{\sigma}$ and temperature deviations per level
$T^{\prime}_{\sigma} = T_\sigma - \bar{T}_{\sigma}$. The resulting
equations are:

\begin{align}
\begin{split}
&\pd{\zeta}{t} =-\nabla\times\left((\zeta + f)\mathbf k\times\mathbf u
    +\dot\sigma\frac{\partial\mathbf u}{\partial\sigma} + RT^\prime\nabla\log p_s \right) \\
&\pd{\delta}{t} =-\nabla\cdot\left((\zeta + f)\mathbf k\times\mathbf u
    +\dot\sigma\frac{\partial\mathbf u}{\partial\sigma} + RT^\prime\nabla\log p_s \right)
    -\nabla^2\left(\frac{||\mathbf u||^2}{2} + \Phi + R \bar{T}\log p_s \right) \\
&\pd{T}{t} =-\mathbf u\cdot\nabla T -\dot\sigma \frac{\partial T}{\partial \sigma}
    +\frac{\kappa T\omega}{p} =-\nabla\cdot \mathbf u T^\prime + T^\prime\delta
    -\dot\sigma \frac{\partial T}{\partial \sigma} +\frac{\kappa T\omega}{p} \\
&\pd{q_i}{t} =-\nabla\cdot \mathbf{u}q_{i} + q_{i}\delta -\dot{\sigma}\pd{q_{i}}{\sigma}  \\
&\pd{\log p_s}{t} =-\frac{1}{p_s}\int_0^1\nabla\cdot(\mathbf up_s)\,d\sigma
    =-\int_0^1\left(\delta +\mathbf u\cdot\nabla\log p_s\right)\,d\sigma
\label{eq:primitive_equations}
\end{split}
\end{align}
with horizontal velocity vector $\mathbf{u}=\nabla(\Delta^{-1}\delta)
+ \mathbf{k} \times\nabla(\Delta^{-1}\zeta)$, Coriolis parameter $f$,
upward-directed unit vector parallel to the z-axis $\mathbf{k}$, ideal
gas constant $R$, heat capacity at constant pressure $C_{p}$, $\kappa=
\frac{R}{C_{p}}$, diagnosed vertical velocity in sigma coordinates
$\dot{\sigma}$, diagnosed change in pressure of a fluid parcel $\omega
\equiv \frac{dp}{dt}$, diagnosed geopotential $\Phi$, diagnosed
virtual temperature $T_{\nu}$ and each moisture species denoted as
$q_{i}$.

Diagnostic quantities are computed as follows:
\begin{align}
    \dot\sigma_{k + \frac{1}{2}} &= -\sigma_{k + \frac{1}{2}}\frac{\partial\log p_s}{\partial t} -\frac{1}{p_s}\int_0^{\sigma_{k + \frac{1}{2}}} \nabla\cdot(p_s\mathbf u)\, d\sigma \\
\frac{\omega_k}{p_s\sigma_k}
&= \mathbf u_k\cdot\nabla \log p_s
-\frac{1}{\sigma_k}\int_0^{\sigma_k}\left(\delta + \mathbf u\cdot\nabla\log p_s \right)\,d\sigma \\
    \Phi_k &= \Phi_{s} + R\int_{\log \sigma_k}^{0} T_{\nu}\,d\log\sigma \label{apx:eq:diagnostic_variables} \\
    T_{\nu} &= T(1 + \left(\frac{R_{vap}}{R} - 1 \right)q - q_{c_{i}} - q_{c_{l}})
\end{align}
where $\Phi_{s}=gz_{s}$ is the geopotential at the surface.

> Our choice of the numerical schemes for interpolation, integrals and
> diagnostics exactly follows Durran’s book [3] §8.6, with the
> addition of moisture species (which are advected by the wind and
> only affect the dynamics through through their effect on the virtual
> temperature). We use semi-implicit time-integration scheme, where
> all right hand side terms are separated into groups that are treated
> either explicitly or implicitly. This avoids severe time step
> limitations due to fast moving gravity waves
