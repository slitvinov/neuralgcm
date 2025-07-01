#!/bin/sh

TF_CPP_MIN_LOG_LEVEL=3 export TF_CPP_MIN_LOG_LEVEL
CUDA_VISIBLE_DEVICES=3 COVERAGE_FILE=.coverage.2 python -m coverage run weather.py

python -m coverage combine
python -m coverage html
