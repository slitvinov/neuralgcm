#!/bin/bash

set -xe

python -c '
from datetime import datetime, timedelta
fmt = "%Y%m%dT%H"
c = datetime.strptime("19900501T00", fmt)
while True:
    print(c.strftime(fmt))
    c += timedelta(hours=1)
' | xargs -P 4 -n 2 --process-slot-var I sh -c '
     dir=$0
     set -- 0-15 16-31 32-48 49-63
     export XLA_PYTHON_CLIENT_PREALLOCATE=false
     export TF_CPP_MIN_LOG_LEVEL=3
     export CUDA_VISIBLE_DEVICES=$I
     mkdir -p $dir
     cd $dir
     date > start
     python ../weather.py $dir 2>stderr 1>stdout
     date > end
     echo $? > status
     python ../../poc/vis.py out.*.raw 2>stderr.vis 1>stdout.vis
     rm -rf *.raw
'
