## Install

```
python -m venv ~/.venv/neuralgcm
. ~/.venv/neuralgcm/bin/activate
python -m pip install jax[cuda12] gcsfs matplotlib tree_math xarray zarr
```

## Test

Save reference raw files

```
for i in *.raw; do cmp $i ~/$i; echo $i $?; done
```

## Data

https://github.com/google-research/arco-era5

The data can be accessed using `gsutil`. Here are some examples:

```
$ gsutil ls gs://gcp-public-data-arco-era5/raw
gs://gcp-public-data-arco-era5/raw/ERA5GRIB/
gs://gcp-public-data-arco-era5/raw/date-variable-pressure_level/
gs://gcp-public-data-arco-era5/raw/date-variable-single_level/
gs://gcp-public-data-arco-era5/raw/date-variable-static/
gs://gcp-public-data-arco-era5/raw/temp/
```

```
$ gsutil ls gs://gcp-public-data-arco-era5/raw/date-variable-pressure_level/1990/05/01
gs://gcp-public-data-arco-era5/raw/date-variable-pressure_level/1990/05/01/fraction_of_cloud_cover/
gs://gcp-public-data-arco-era5/raw/date-variable-pressure_level/1990/05/01/geopotential/
gs://gcp-public-data-arco-era5/raw/date-variable-pressure_level/1990/05/01/ozone_mass_mixing_ratio/
gs://gcp-public-data-arco-era5/raw/date-variable-pressure_level/1990/05/01/potential_vorticity/
gs://gcp-public-data-arco-era5/raw/date-variable-pressure_level/1990/05/01/specific_cloud_ice_water_content/
gs://gcp-public-data-arco-era5/raw/date-variable-pressure_level/1990/05/01/specific_cloud_liquid_water_content/
gs://gcp-public-data-arco-era5/raw/date-variable-pressure_level/1990/05/01/specific_humidity/
gs://gcp-public-data-arco-era5/raw/date-variable-pressure_level/1990/05/01/temperature/
gs://gcp-public-data-arco-era5/raw/date-variable-pressure_level/1990/05/01/u_component_of_wind/
gs://gcp-public-data-arco-era5/raw/date-variable-pressure_level/1990/05/01/v_component_of_wind/
gs://gcp-public-data-arco-era5/raw/date-variable-pressure_level/1990/05/01/vertical_velocity/
```

```
$ gsutil ls gs://gcp-public-data-arco-era5/raw/date-variable-pressure_level/1990/05/01/u_component_of_wind
gs://gcp-public-data-arco-era5/raw/date-variable-pressure_level/1990/05/01/u_component_of_wind/1.nc
gs://gcp-public-data-arco-era5/raw/date-variable-pressure_level/1990/05/01/u_component_of_wind/10.nc
gs://gcp-public-data-arco-era5/raw/date-variable-pressure_level/1990/05/01/u_component_of_wind/100.nc
gs://gcp-public-data-arco-era5/raw/date-variable-pressure_level/1990/05/01/u_component_of_wind/1000.nc
gs://gcp-public-data-arco-era5/raw/date-variable-pressure_level/1990/05/01/u_component_of_wind/125.nc
gs://gcp-public-data-arco-era5/raw/date-variable-pressure_level/1990/05/01/u_component_of_wind/150.nc
...
```

```
$ gsutil ls -lh gs://gcp-public-data-arco-era5/raw/ERA5GRIB/HRES/Daily/1990/19900501*
  5.04 GiB  2022-10-03T04:44:02Z  gs://gcp-public-data-arco-era5/raw/ERA5GRIB/HRES/Daily/1990/19900501_hres_dve.grb2
 11.25 GiB  2022-10-03T04:49:02Z  gs://gcp-public-data-arco-era5/raw/ERA5GRIB/HRES/Daily/1990/19900501_hres_o3q.grb2
   3.5 GiB  2022-10-03T04:44:02Z  gs://gcp-public-data-arco-era5/raw/ERA5GRIB/HRES/Daily/1990/19900501_hres_qrqs.grb2
  5.04 GiB  2022-10-03T04:49:47Z  gs://gcp-public-data-arco-era5/raw/ERA5GRIB/HRES/Daily/1990/19900501_hres_tw.grb2
TOTAL: 4 objects, 26664833880 bytes (24.83 GiB)
```

dve: Divergence and vorticity at various levels
o3q: Ozone and specific humidity-related variables
qrqs: Cloud and precipitation-related variables like rain/snow mixing ratios
tw: Temperature and wind components

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
- https://old.wmo.int/extranet/pages/prog/www/WMOCodes/Guides/GRIB/GRIB2_062006.pdf
- https://github.com/ecmwf/cfgrib
- https://get.ecmwf.int/repository/test-data/cfgrib/era5-levels-members.grib
- https://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_doc

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
