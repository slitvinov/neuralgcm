gsutil -m cp -r gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3  .
gsutil -m cp -r gs://gcp-public-data-arco-era5/ar/model-level-1h-0p25deg.zarr-v1 .

```
$ gsutil ls gs://gcp-public-data-arco-era5/raw
gs://gcp-public-data-arco-era5/raw/ERA5GRIB/
gs://gcp-public-data-arco-era5/raw/date-variable-pressure_level/
gs://gcp-public-data-arco-era5/raw/date-variable-single_level/
gs://gcp-public-data-arco-era5/raw/date-variable-static/
gs://gcp-public-data-arco-era5/raw/temp/
```

```
gsutil ls gs://gcp-public-data-arco-era5/raw/date-variable-pressure_level/1990/05/01
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
gsutil ls gs://gcp-public-data-arco-era5/raw/date-variable-pressure_level/1990/05/01/u_component_of_wind
gs://gcp-public-data-arco-era5/raw/date-variable-pressure_level/1990/05/01/u_component_of_wind/1.nc
gs://gcp-public-data-arco-era5/raw/date-variable-pressure_level/1990/05/01/u_component_of_wind/10.nc
gs://gcp-public-data-arco-era5/raw/date-variable-pressure_level/1990/05/01/u_component_of_wind/100.nc
gs://gcp-public-data-arco-era5/raw/date-variable-pressure_level/1990/05/01/u_component_of_wind/1000.nc
gs://gcp-public-data-arco-era5/raw/date-variable-pressure_level/1990/05/01/u_component_of_wind/125.nc
gs://gcp-public-data-arco-era5/raw/date-variable-pressure_level/1990/05/01/u_component_of_wind/150.nc
...
```
