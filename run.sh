#!/bin/sh

TF_CPP_MIN_LOG_LEVEL=3 export TF_CPP_MIN_LOG_LEVEL
CUDA_VISIBLE_DEVICES=0 python baroclinic_instability.py &
CUDA_VISIBLE_DEVICES=1 python held_suarez.py &
CUDA_VISIBLE_DEVICES=2 python weather_forecast_on_era5.py &
wait
