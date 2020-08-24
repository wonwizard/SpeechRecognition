#!/bin/sh

export CUDA_VISIBLE_DEVICES="0"

<<NO_CONFIG
BATCH_SIZE=4
WORKER_SIZE=1
MAX_EPOCHS=10

python ./main.py --batch_size $BATCH_SIZE --workers $WORKER_SIZE --use_attention --max_epochs $MAX_EPOCHS
NO_CONFIG

python ./main.py
