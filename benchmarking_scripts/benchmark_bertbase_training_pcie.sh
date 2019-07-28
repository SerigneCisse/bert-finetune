#!/usr/bin/env bash

echo -e "\n[INFO ] Caching Data\n"

python3 cache_bertbase.py
python3 cache_data.py

echo -e "\n[INFO ] Running on $N_GPU GPUs!\n"

echo -e "\n[INFO ] Optimized, efficient comms\n"

mpirun -np $N_GPU -H localhost:$N_GPU \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -x NCCL_P2P_DISABLE=1 \
    -mca pml ob1 -mca btl ^openib \
    python3 news_classification.py \
    --xla --amp --sparse_as_dense --fp16_allreduce --lazyadam \
    --batch_size 25 --lr 5e-5 --epochs 3 --maxseqlen 512 \
    --results /results/optimized_fastest/
