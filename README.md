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
