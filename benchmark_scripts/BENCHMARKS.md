# Benchmarks

* Here are the benchmarks to reproduce the figures shown in the diagrams.
* Scripts should be copied into the root directory of the repository and run from here.

## Introduction

**BERTBASE**

* Slowest Baseline (16GB): [`benchmark_bertbase_baseline.sh`](benchmark_bertbase_baseline.sh)
* Fastest V100 (16GB): [`benchmark_bertbase_fastest.sh`](benchmark_bertbase_fastest.sh)
* Fastest V100 (32GB): [`benchmark_bertbase_fastest_32.sh`](benchmark_bertbase_fastest_32.sh)

**BERTLARGE**

* Slowest Baseline (16GB): [`benchmark_bertlarge_baseline.sh`](benchmark_bertlarge_baseline.sh)
* Fastest V100 (16GB): [`benchmark_bertlarge_fastest.sh`](benchmark_bertlarge_fastest.sh)
* Fastest V100 (32GB): [`benchmark_bertlarge_fastest_32.sh`](benchmark_bertlarge_fastest_32.sh)

## TF Performance Features

**BERTBASE**

* V100 16GB: [`benchmark_bertbase_tf.sh`](benchmark_bertbase_tf.sh)
* V100 32GB: [`benchmark_bertbase_tf_32.sh`](benchmark_bertbase_tf_32.sh)

**BERTLARGE**

* V100 16GB: [`benchmark_bertlarge_tf.sh`](benchmark_bertlarge_tf.sh)
* V100 32GB: [`benchmark_bertlarge_tf_32.sh`](benchmark_bertlarge_tf_32.sh)

## Horovod Performance Features

**BERTBASE**

* V100 16GB: [`benchmark_bertbase_hvd.sh`](benchmark_bertbase_hvd.sh)
* V100 32GB: [`benchmark_bertbase_hvd_32.sh`](benchmark_bertbase_hvd_32.sh)

**BERTLARGE**

* V100 16GB: [`benchmark_bertlarge_hvd.sh`](benchmark_bertlarge_hvd.sh)
* V100 32GB: [`benchmark_bertlarge_hvd_32.sh`](benchmark_bertlarge_hvd_32.sh)

## Multi-GPU Scaling Performance

* BERTBASE NVLINK Test: [`benchmark_bertbase_no_nvlink_32.sh`](benchmark_bertbase_no_nvlink_32.sh)
* BERTLARGE NVLINK Test: [`benchmark_bertlarge_no_nvlink_32.sh`](benchmark_bertlarge_no_nvlink_32.sh)
