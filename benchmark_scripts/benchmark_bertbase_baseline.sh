#!/usr/bin/env bash

export TF_ENABLE_AUTO_MIXED_PRECISION=0

echo -e "\n[INFO ] Caching Data\n"

python3 cache_data.py --agnews --bertbase

echo -e "\n[INFO ] Running on $N_GPU GPUs!\n"

mpirun -np $N_GPU -H localhost:$N_GPU \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=WARNING -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python3 news_classification.py \
    --epochs 3 --maxseqlen 512 --batch_size 16 --results /results/
