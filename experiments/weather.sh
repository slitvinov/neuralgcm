#!/bin/bash

set -x
set -e
c=19900501T00
while :
do echo $c
   c=$(date -d "${c:0:8} ${c:9:2}:00 +1 hour" '+%Y%m%dT%H')
done | xargs -P 4 -n 2 --process-slot-var I sh -c '
     export XLA_PYTHON_CLIENT_PREALLOCATE=false
     export TF_CPP_MIN_LOG_LEVEL=3
     export CUDA_VISIBLE_DEVICES=$I
     mkdir -p $0
     cd $0
     date > start
     python ../weather.py $0 2>stderr 1>stdout
     date > end
     echo $? > status
     exec python ../poc/vis.py out.*.raw 2>stderr.vis 1>stdout.vis
'
