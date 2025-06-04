## Install

```
python -m venv ~/.venv/neuralgcm
. ~/.venv/neuralgcm/bin/activate
python -m pip install jax[cuda12] gcsfs matplotlib pint tree_math xarray zarr
python -m pip install coverage
```

```
git clone git@github.com:neuralgcm/neuralgcm
git clone git@github.com:neuralgcm/dinosaur
git clone git@github.com:google-deepmind/dm-haiku
git clone git@github.com:google-research/weatherbench2
```

## Regression Testing with Raw Data

The primary simulation scripts have been updated to output raw numerical data, which can be used for regression testing purposes.

- `baroclinic_instability.py` now generates `b.09.raw`.
- `held_suarez.py` now generates `h.12.raw`.
- `weather_forecast_on_era5.py` now generates `w.00.raw`, `w.01.raw`, and `w.02.raw`.

A basic test script, `test_data_generation.py`, is provided to facilitate this.
To use it:
1. Ensure you have Python 3 installed.
2. Run the script from the root of the repository: `python3 test_data_generation.py`

This script will execute the three main simulation scripts and verify that the corresponding `.raw` files are created in the root directory.

**Initial Setup for Golden Files:**
To establish a baseline for regression tests:
1. Run each of the main simulation scripts once (e.g., `python3 baroclinic_instability.py`, etc.) to generate the initial set of `.raw` files.
2. Carefully review these files to ensure their contents are correct and represent a "golden" state.
3. Commit these `.raw` files to your repository.

Future enhancements to `test_data_generation.py` could include comparing newly generated `.raw` files against these committed golden files to detect regressions.
