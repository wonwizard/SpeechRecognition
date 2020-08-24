#!/bin/sh

export CUDA_VISIBLE_DEVICES="0"

BATCH_SIZE=10
WORKER_SIZE=1
#MAX_EPOCHS=1000
MAX_EPOCHS=400

python3 ./main.py --batch_size $BATCH_SIZE --workers $WORKER_SIZE --use_attention --max_epochs $MAX_EPOCHS
