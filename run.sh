#!/bin/sh

TF_CPP_MIN_LOG_LEVEL=3 export TF_CPP_MIN_LOG_LEVEL
CUDA_VISIBLE_DEVICES=1 python baroclinic.py &
CUDA_VISIBLE_DEVICES=2 python held_suarez.py &
CUDA_VISIBLE_DEVICES=3 python weather.py &
wait
