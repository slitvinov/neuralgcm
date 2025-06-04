#!/bin/sh

TF_CPP_MIN_LOG_LEVEL=3 export TF_CPP_MIN_LOG_LEVEL
CUDA_VISIBLE_DEVICES=0 python -m coverage run --parallel-mode baroclinic_instability.py &
CUDA_VISIBLE_DEVICES=1 python -m coverage run --parallel-mode held_suarez.py &
CUDA_VISIBLE_DEVICES=2 python -m coverage run --parallel-mode weather_forecast_on_era5.py &
wait

python -m coverage combine
python -m coverage html
