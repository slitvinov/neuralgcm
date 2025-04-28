CUDA_VISIBLE_DEVICE=0 python -m coverage run --parallel-mode baroclinic_instability.py &
CUDA_VISIBLE_DEVICE=1 python -m coverage run --parallel-mode held_suarez.py &
CUDA_VISIBLE_DEVICE=2 python -m coverage run --parallel-mode weather_forecast_on_era5.py &
wait

