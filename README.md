## Install

```
python -m venv ~/.venv/neuralgcm
. ~/.venv/neuralgcm/bin/activate
python -m pip install jax[cuda12] gcsfs matplotlib xarray zarr
```

## Test

Save reference raw files

```
for i in *.raw; do cmp $i ~/$i; echo $i $?; done
```

## Data

- https://github.com/google-research/arco-era5
- https://github.com/NOAA-EMC/NCEPLIBS-g2c
- https://github.com/NOAA-EMC/NCEPLIBS-g2
- https://github.com/weathersource/grib_api

To download raw data (~24Gb)

```
gsutil cp gs://gcp-public-data-arco-era5/raw/ERA5GRIB/HRES/Daily/1990/19900501* ~
```

## References

- [neuralgcm/dinosaur](https://github.com/neuralgcm/dinosaur)
- Kochkov, D., Yuval, J., Langmore, I., Norgaard, P., Smith, J.,
  Mooers, G., ... & Hoyer, S. (2024). Neural general circulation
  models for weather and climate. Nature, 632(8027), 1060-1066. <br>
  [doi:10.1038/s41586-024-07744-y](https://doi.org/10.1038/s41586-024-07744-y)
  [Supplementary
  information](https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-07744-y/MediaObjects/41586_2024_7744_MOESM1_ESM.pdf)
- Durran, D. R. *Numerical methods for fluid dynamics: With
  applications to geophysics*, Second edn, Vol. 32 (Springer, New
  York, 2010)
- Whitaker, J. S. & Kar, S. K. Implicit–explicit runge–kutta methods
  for fast–slow wave problems. Monthly weather review 141, 3426–3434
  (2013)
  [doi:10.1175/MWR-D-13-00132.1](https://doi.org/10.1175/MWR-D-13-00132.1)
- Jablonowski, C., & Williamson, D. L. (2011). The pros and cons of
  diffusion, filters and fixers in atmospheric general circulation
  models. Numerical techniques for global atmospheric models, 381-493.
  [doi:10.1007/978-3-642-11640-7_13](https://doi.org/10.1007/978-3-642-11640-7_13)
- [docs/primitive.pdf](docs/primitive.pdf)
- [docs/durran.pdf](docs/durran.pdf)
- Lynch, P., & Huang, X. Y. (1992). Initialization of the HIRLAM model
  using a digital filter. Monthly Weather Review, 120(6),
  1019-1034.
  [doi:10.1175/1520-0493(1992)120%3C1019:IOTHMU%3E2.0.CO;2]
  (https://doi.org/10.1175/1520-0493(1992)120%3C1019:IOTHMU%3E2.0.CO;2)

> Our choice of the numerical schemes for interpolation, integrals and
> diagnostics exactly follows Durran’s book [3] §8.6, with the
> addition of moisture species (which are advected by the wind and
> only affect the dynamics through through their effect on the virtual
> temperature). We use semi-implicit time-integration scheme, where
> all right hand side terms are separated into groups that are treated
> either explicitly or implicitly. This avoids severe time step
> limitations due to fast moving gravity waves. Rather than the
> traditional semi-implicit leapfrog method, we use implicit-explicit
> Runge-Kutta methods to avoid the complexity of keeping track of
> multiple time-steps and time-filtering required by the traditional
> semi-implicit leapfrog method.  Specifically, we use the
> semi-implicit Lorenz three cycle scheme (SIL3), which was developed
> specifically for global spectral weather
> models~\cite{whitaker2013implicit}.
