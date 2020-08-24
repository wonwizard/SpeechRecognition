#!/bin/sh

GPU_SIZE=1
CPU_SIZE=4
DATASET="sr-hack-2019-50000"

<<NO_CONFIG
BATCH_SIZE=8
WORKER_SIZE=2
MAX_EPOCHS=30

nsml run -g $GPU_SIZE -c $CPU_SIZE -d $DATASET -a "--batch_size $BATCH_SIZE --workers $WORKER_SIZE --use_attention --max_epochs $MAX_EPOCHS"
NO_CONFIG

# nsml run -g 1 -c 4 -d "sr-hack-2019-50000" -a "--config config/legacy/cfg0/baseline.cfg0.json"
echo nsml run -g 3 -c 8 -d "sr-hack-2019-50000" -a "--config $1 $2 $3"
nsml run -g 1 -c 4 -d "sr-hack-2019-50000" -a "--config $1 $2 $3"
