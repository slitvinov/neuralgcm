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

- [neuralgcm/dinosaur](https://github.com/neuralgcm/dinosaur)
- Kochkov, D., Yuval, J., Langmore, I., Norgaard, P., Smith, J., Mooers,
G., ... & Hoyer, S. (2024). Neural general circulation models for
weather and climate. Nature, 632(8027), 1060-1066.
[doi:10.1038/s41586-024-07744-y](https://doi.org/10.1038/s41586-024-07744-y)
- [Supplementary information](https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-07744-y/MediaObjects/41586_2024_7744_MOESM1_ESM.pdf)
- Durran, D. R. *Numerical methods for fluid dynamics: With applications to geophysics*, Second edn, Vol. 32 (Springer, New York, 2010)
- Whitaker, J. S. & Kar, S. K. Implicit–explicit runge–kutta methods
for fast–slow wave problems. Monthly weather review 141, 3426–3434
(2013)
[doi:10.1175/MWR-D-13-00132.1](https://doi.org/10.1175/MWR-D-13-00132.1)
- [docs/primitive.pdf](docs/primitive.pdf)
- [docs/durran.pdf](docs/durran.pdf)

> Our choice of the numerical schemes for interpolation, integrals and
> diagnostics exactly follows Durran’s book [3] §8.6, with the
> addition of moisture species (which are advected by the wind and
> only affect the dynamics through through their effect on the virtual
> temperature). We use semi-implicit time-integration scheme, where
> all right hand side terms are separated into groups that are treated
> either explicitly or implicitly. This avoids severe time step
> limitations due to fast moving gravity waves
