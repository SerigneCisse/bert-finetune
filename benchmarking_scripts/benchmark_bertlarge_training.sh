#!/usr/bin/env bash

echo -e "\n[INFO ] Caching Data\n"

python3 cache_bertlarge.py
python3 cache_data.py

echo -e "\n[INFO ] Running on $N_GPU GPUs!\n"

echo -e "\n[INFO ] Baseline, slowest\n"

mpirun -np $N_GPU -H localhost:$N_GPU \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=WARNING -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python3 news_classification.py \
    --batch_size 2 --bertlarge --lr 5e-5 --epochs 3 --maxseqlen 512 \
    --results /results/baseline/

echo -e "\n[INFO ] Baseline, fp16_allreduce\n"

mpirun -np $N_GPU -H localhost:$N_GPU \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=WARNING -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python3 news_classification.py \
    --fp16_allreduce \
    --batch_size 2 --bertlarge --lr 5e-5 --epochs 3 --maxseqlen 512 \
    --results /results/baseline_fp16/

echo -e "\n[INFO ] Baseline, sparse_as_dense\n"

mpirun -np $N_GPU -H localhost:$N_GPU \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=WARNING -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python3 news_classification.py \
    --sparse_as_dense \
    --batch_size 2 --bertlarge --lr 5e-5 --epochs 3 --maxseqlen 512 \
    --results /results/baseline_nosparse/

echo -e "\n[INFO ] Baseline, efficient comms\n"

mpirun -np $N_GPU -H localhost:$N_GPU \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=WARNING -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python3 news_classification.py \
    --sparse_as_dense --fp16_allreduce \
    --batch_size 2 --bertlarge --lr 5e-5 --epochs 3 --maxseqlen 512 \
    --results /results/baseline_efficient/

echo -e "\n[INFO ] Optimized, inefficient comms\n"

mpirun -np $N_GPU -H localhost:$N_GPU \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=WARNING -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python3 news_classification.py \
    --xla --amp \
    --batch_size 2 --bertlarge --lr 5e-5 --epochs 3 --maxseqlen 512 \
    --results /results/optimized_slow/

echo -e "\n[INFO ] Optimized, fp16_allreduce\n"

mpirun -np $N_GPU -H localhost:$N_GPU \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=WARNING -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python3 news_classification.py \
    --xla --amp --fp16_allreduce \
    --batch_size 3 --bertlarge --lr 5e-5 --epochs 3 --maxseqlen 512 \
    --results /results/optimized_fp16/

echo -e "\n[INFO ] Optimized, sparse_as_dense comms\n"

mpirun -np $N_GPU -H localhost:$N_GPU \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=WARNING -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python3 news_classification.py \
    --xla --amp --sparse_as_dense \
    --batch_size 2 --bertlarge --lr 5e-5 --epochs 3 --maxseqlen 512 \
    --results /results/optimized_nosparse/

echo -e "\n[INFO ] Optimized, efficient comms\n"

mpirun -np $N_GPU -H localhost:$N_GPU \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=WARNING -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python3 news_classification.py \
    --xla --amp --sparse_as_dense --fp16_allreduce \
    --batch_size 3 --bertlarge --lr 5e-5 --epochs 3 --maxseqlen 512 \
    --results /results/optimized_fastest/

echo -e "\n[INFO ] Optimized, XLA only\n"

mpirun -np $N_GPU -H localhost:$N_GPU \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=WARNING -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python3 news_classification.py \
    --xla --sparse_as_dense --fp16_allreduce \
    --batch_size 2 --bertlarge --lr 5e-5 --epochs 3 --maxseqlen 512 \
    --results /results/optimized_xla_only/

echo -e "\n[INFO ] Optimized, AMP only\n"

mpirun -np $N_GPU -H localhost:$N_GPU \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=WARNING -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python3 news_classification.py \
    --amp --sparse_as_dense --fp16_allreduce \
    --batch_size 3 --bertlarge --lr 5e-5 --epochs 3 --maxseqlen 512 \
    --results /results/optimized_amp_only/
