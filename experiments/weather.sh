#!/bin/bash

set -xe

python -c '
from datetime import datetime, timedelta
fmt = "%Y%m%dT%H"
c = datetime.strptime("20000101T00", fmt)
for i in range(50000):
    print(c.strftime(fmt))
    c += timedelta(hours=1)
' | xargs -P 4 -n 1 --process-slot-var I sh -xc '
     dir=$0
     set -- 0-31 32-63 64-95 96-127
     shift $I
     cpu=$1
     export XLA_PYTHON_CLIENT_PREALLOCATE=false
     export TF_CPP_MIN_LOG_LEVEL=3
     export CUDA_VISIBLE_DEVICES=$I
     mkdir -p $dir
     cd $dir
     date > start
     taskset -c $cpu python -u ../weather.py $dir 2>stderr 1>stdout
     date > end
     echo $? > status
     taskset -c $cpu python -u ../../poc/vis.py out.*.raw 2>stderr.vis 1>stdout.vis
     taskset -c $cpu python -u ../../poc/contour.py out.*.raw 2>stderr.contour 1>stdout.contour
     rm -rf *.raw
'
