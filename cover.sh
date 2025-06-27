#!/bin/sh

TF_CPP_MIN_LOG_LEVEL=3 export TF_CPP_MIN_LOG_LEVEL
CUDA_VISIBLE_DEVICES=1 COVERAGE_FILE=.coverage.0 python -m coverage run baroclinic.py &
CUDA_VISIBLE_DEVICES=2 COVERAGE_FILE=.coverage.1 python -m coverage run held_suarez.py &
CUDA_VISIBLE_DEVICES=3 COVERAGE_FILE=.coverage.2 python -m coverage run weather.py &
wait

python -m coverage combine &&
python -m coverage html
