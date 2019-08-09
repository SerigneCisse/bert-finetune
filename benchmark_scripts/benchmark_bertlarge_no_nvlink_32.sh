#!/usr/bin/env bash

echo -e "\n[INFO ] Caching Data\n"

python3 cache_data.py --agnews --bertlarge

echo -e "\n[INFO ] Running on $N_GPU GPUs!\n"

mpirun -np $N_GPU -H localhost:$N_GPU \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    -x NCCL_P2P_DISABLE=1 \
    python3 news_classification.py \
    --xla --amp --sparse_as_dense --fp16_allreduce \
    --batch_size 16 --epochs 3 --maxseqlen 512 \
    --results /results/ --bertlarge --lazyadam