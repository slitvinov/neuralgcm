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
     export XLA_PYTHON_CLIENT_PREALLOCATE=false
     export TF_CPP_MIN_LOG_LEVEL=3
     export CUDA_VISIBLE_DEVICES=$I
     mkdir -p $0
     cd $0
     date > start
     python ../weather.py $0 2>stderr 1>stdout
     date > end
     echo $? > status
     python ../vis.py out.*.raw 2>stderr.vis 1>stdout.vis
     rm -rf *.raw
'
