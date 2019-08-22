#!/usr/bin/env bash

echo -e "\n[INFO ] Caching Data\n"

python3 cache_data.py --bertbase

echo -e "\n[INFO ] Running on $N_GPU GPUs!\n"

echo -e "\n[INFO ] LR == 0.00001\n"

mpirun -np $N_GPU -H localhost:$N_GPU \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=WARNING -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python3 dbpedia_classification.py \
    --xla --amp --sparse_as_dense --fp16_allreduce \
    --epochs 4 --lr "0.00001" --radam

echo -e "\n[INFO ] LR == 0.00002\n"

mpirun -np $N_GPU -H localhost:$N_GPU \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=WARNING -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python3 dbpedia_classification.py \
    --xla --amp --sparse_as_dense --fp16_allreduce \
    --epochs 4 --lr "0.00002" --radam

echo -e "\n[INFO ] LR == 0.00003\n"

mpirun -np $N_GPU -H localhost:$N_GPU \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=WARNING -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python3 dbpedia_classification.py \
    --xla --amp --sparse_as_dense --fp16_allreduce \
    --epochs 4 --lr "0.00003" --radam

echo -e "\n[INFO ] LR == 0.00004\n"

mpirun -np $N_GPU -H localhost:$N_GPU \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=WARNING -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python3 dbpedia_classification.py \
    --xla --amp --sparse_as_dense --fp16_allreduce \
    --epochs 4 --lr "0.00004" --radam

echo -e "\n[INFO ] LR == 0.00005\n"

mpirun -np $N_GPU -H localhost:$N_GPU \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=WARNING -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python3 dbpedia_classification.py \
    --xla --amp --sparse_as_dense --fp16_allreduce \
    --epochs 4 --lr "0.00005" --radam

echo -e "\n[INFO ] LR == 0.00006\n"

mpirun -np $N_GPU -H localhost:$N_GPU \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=WARNING -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python3 dbpedia_classification.py \
    --xla --amp --sparse_as_dense --fp16_allreduce \
    --epochs 4 --lr "0.00006" --radam
    
echo -e "\n[INFO ] LR == 0.00007\n"

mpirun -np $N_GPU -H localhost:$N_GPU \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=WARNING -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python3 dbpedia_classification.py \
    --xla --amp --sparse_as_dense --fp16_allreduce \
    --epochs 4 --lr "0.00007" --radam
    
echo -e "\n[INFO ] LR == 0.00008\n"

mpirun -np $N_GPU -H localhost:$N_GPU \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=WARNING -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python3 dbpedia_classification.py \
    --xla --amp --sparse_as_dense --fp16_allreduce \
    --epochs 4 --lr "0.00008" --radam