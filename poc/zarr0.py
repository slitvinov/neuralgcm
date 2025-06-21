import xarray


def open_era5(path, time):
    ds = xarray.open_zarr(path,
                          chunks=None,
                          storage_options={
                              "token": "anon",
                              "cache_storage": "."
                          })
    return ds.sel(time=time)


ds_arco_era5 = xarray.merge([
    open_era5(
        "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
        time="19900501T00",
    ).drop_dims("level"),
    open_era5(
        "gs://gcp-public-data-arco-era5/ar/model-level-1h-0p25deg.zarr-v1",
        time="19900501T00",
    ),
])

ds = ds_arco_era5[[
    "u_component_of_wind",
    "v_component_of_wind",
    "temperature",
    "specific_humidity",
    "specific_cloud_liquid_water_content",
    "specific_cloud_ice_water_content",
    "surface_pressure",
]]
