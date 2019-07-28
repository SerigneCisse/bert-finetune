#!/usr/bin/env bash

echo -e "\n[INFO ] Caching Data\n"

python3 cache_bertlarge.py
python3 cache_data.py

echo -e "\n[INFO ] Running on $N_GPU GPUs!\n"

echo -e "\n[INFO ] Optimized, efficient comms\n"

mpirun -np $N_GPU -H localhost:$N_GPU \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=WARNING -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python3 news_classification.py \
    --xla --amp --sparse_as_dense --fp16_allreduce \
    --batch_size 16 --bertlarge --lr 5e-5 --epochs 3 --maxseqlen 512 \
    --results /results/optimized_fastest/

echo -e "\n[INFO ] Optimized, XLA only\n"

mpirun -np $N_GPU -H localhost:$N_GPU \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=WARNING -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python3 news_classification.py \
    --xla --sparse_as_dense --fp16_allreduce \
    --batch_size 8 --bertlarge --lr 5e-5 --epochs 3 --maxseqlen 512 \
    --results /results/optimized_xla_only/

echo -e "\n[INFO ] Optimized, AMP only\n"

mpirun -np $N_GPU -H localhost:$N_GPU \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=WARNING -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python3 news_classification.py \
    --amp --sparse_as_dense --fp16_allreduce \
    --batch_size 16 --bertlarge --lr 5e-5 --epochs 3 --maxseqlen 512 \
    --results /results/optimized_amp_only/

echo -e "\n[INFO ] Baseline, efficient comms\n"

mpirun -np $N_GPU -H localhost:$N_GPU \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=WARNING -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python3 news_classification.py \
    --sparse_as_dense --fp16_allreduce \
    --batch_size 8 --bertlarge --lr 5e-5 --epochs 3 --maxseqlen 512 \
    --results /results/baseline_efficient/
