#!/usr/bin/env bash

echo -e "\n[INFO ] Caching Data\n"

python3 cache_data.py --agnews --bertlarge

echo -e "\n[INFO ] Running on $N_GPU GPUs!\n"

export TF_ENABLE_AUTO_MIXED_PRECISION=0

echo -e "\n[INFO ] Baseline (efficient comms)\n"

mpirun -np $N_GPU -H localhost:$N_GPU \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=WARNING -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python3 news_classification.py \
    --sparse_as_dense --fp16_allreduce \
    --epochs 3 --maxseqlen 512 --batch_size 2 --bertlarge

echo -e "\n[INFO ] XLA\n"

mpirun -np $N_GPU -H localhost:$N_GPU \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=WARNING -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python3 news_classification.py \
    --xla --sparse_as_dense --fp16_allreduce \
    --epochs 3 --maxseqlen 512 --batch_size 2 --bertlarge
    
export TF_ENABLE_AUTO_MIXED_PRECISION=1
    
echo -e "\n[INFO ] AMP\n"

mpirun -np $N_GPU -H localhost:$N_GPU \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=WARNING -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python3 news_classification.py \
    --amp --sparse_as_dense --fp16_allreduce \
    --epochs 3 --maxseqlen 512 --batch_size 3 --bertlarge
    
echo -e "\n[INFO ] Fastest\n"

mpirun -np $N_GPU -H localhost:$N_GPU \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=WARNING -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python3 news_classification.py \
    --xla --amp --sparse_as_dense --fp16_allreduce \
    --epochs 3 --maxseqlen 512 --batch_size 3 --bertlarge
