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

```
$ python gaussian.py ~/19900501_hres_qrqs.grb2_.gstmp  | grep 'PARAMETER\[' | sort -g | uniq -c
     71 PARAMETER[parameter_category][parameter_number]='Specific Rain Water Content'
     71 PARAMETER[parameter_category][parameter_number]='Specific Snow Water Content'
$ python -u grib2.py ~/19900501_hres_tw.grb2_.gstmp | grep 'PARAMETER\[' | sort -g | uniq -c
      2 PARAMETER[parameter_category][parameter_number]='Temperature'
      1 PARAMETER[parameter_category][parameter_number]='Vertical Velocity (Pressure)'
$ python -u grib2.py ~/19900501_hres_dve.grb2_.gstmp | grep 'PARAMETER\[' | sort -g | uniq -c
      1 PARAMETER[parameter_category][parameter_number]='Relative Divergence'
      2 PARAMETER[parameter_category][parameter_number]='Relative Vorticity'
```


```
gsutil ls gs://gcp-public-data-arco-era5/raw/date-variable-single_level/1990/05/01/surface_pressure/
gs://gcp-public-data-arco-era5/raw/date-variable-single_level/1990/05/01/surface_pressure/surface.nc
```

- https://old.wmo.int/extranet/pages/prog/www/WMOCodes/Guides/GRIB/GRIB2_062006.pdf
- https://github.com/ecmwf/cfgrib
- https://get.ecmwf.int/repository/test-data/cfgrib/era5-levels-members.grib
- https://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_doc
