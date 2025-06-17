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
