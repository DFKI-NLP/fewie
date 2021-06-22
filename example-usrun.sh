#!/bin/sh
IMAGE=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.02-py3.sqsh
WORKDIR=/netscratch/ychen/fewie

srun -K \
  --container-mounts=/netscratch:/netscratch,/ds:/ds,$HOME:$HOME \
  --container-workdir=$WORKDIR \
  --container-image=$IMAGE \
  --ntasks=1 \
  --nodes=1 \
  pip3 install . && \
  python evaluate.py
