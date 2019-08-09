#!/usr/bin/env bash

echo -e "\n[INFO ] Caching Data\n"

python3 cache_data.py --agnews --bertbase

echo -e "\n[INFO ] Running on $N_GPU GPUs!\n"

echo -e "\n[INFO ] Slowest comms\n"

mpirun -np $N_GPU -H localhost:$N_GPU \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=WARNING -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python3 news_classification.py \
    --xla --amp \
    --epochs 3 --maxseqlen 512 --batch_size 12
    
echo -e "\n[INFO ] sparse_as_dense comms\n"
    
mpirun -np $N_GPU -H localhost:$N_GPU \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=WARNING -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python3 news_classification.py \
    --xla --amp --sparse_as_dense \
    --epochs 3 --maxseqlen 512 --batch_size 12
    
echo -e "\n[INFO ] fp16 comms\n"
    
mpirun -np $N_GPU -H localhost:$N_GPU \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=WARNING -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python3 news_classification.py \
    --xla --amp --fp16_allreduce \
    --epochs 3 --maxseqlen 512 --batch_size 12
    
echo -e "\n[INFO ] Optimized comms\n"
    
mpirun -np $N_GPU -H localhost:$N_GPU \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=WARNING -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python3 news_classification.py \
    --xla --amp --sparse_as_dense --fp16_allreduce \
    --epochs 3 --maxseqlen 512 --batch_size 23
