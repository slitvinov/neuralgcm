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
- applications to geophysics Second edn, Vol. 32 (Springer, New York,
- 2010)

> Our choice of the numerical schemes for interpolation, integrals and
> diagnostics exactly follows Durran’s book [3] §8.6, with the
> addition of moisture species (which are advected by the wind and
> only affect the dynamics through through their effect on the virtual
> temperature). We use semi-implicit time-integration scheme, where
> all right hand side terms are separated into groups that are treated
> either explicitly or implicitly. This avoids severe time step
> limitations due to fast moving gravity waves
