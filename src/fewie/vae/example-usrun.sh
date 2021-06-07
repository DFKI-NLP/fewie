#!/bin/sh
username="$USER"
IMAGE=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.02-py3.sqsh
WORKDIR=/netscratch/$username/code/fewie/src/fewie/vae

srun -K \
  --container-mounts=/netscratch:/netscratch,/ds:/ds,$HOME:$HOME \
  --container-workdir=$WORKDIR \
  --container-image=$IMAGE \
  --ntasks=1 \
  --nodes=1 \
  $*
