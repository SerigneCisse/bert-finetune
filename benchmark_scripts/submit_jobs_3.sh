#!/usr/bin/env bash

# for internal use

# HVD tests

ngc batch run --name "Benchmark BERTBASE HVD 16GB" \
--preempt RUNONCE --total-runtime 100000s --image "nvidian/sae/ai-lab:19.08-tf" --ace nv-us-west-2 --instance dgx1v.16g.8.norm --commandline \
"bash -c 'export N_GPU=8 && git clone --depth 1 https://github.com/NVAITC/bert-finetune && cd bert-finetune && cp benchmark_scripts/benchmark_bertbase_hvd.sh run_script.sh && bash run_script.sh'" \
--result /results --org nvidian --team sae

ngc batch run --name "Benchmark BERTBASE HVD 32GB" \
--preempt RUNONCE --total-runtime 100000s --image "nvidian/sae/ai-lab:19.08-tf" --ace nv-us-west-2 --instance dgx1v.32g.8.norm --commandline \
"bash -c 'export N_GPU=8 && git clone --depth 1 https://github.com/NVAITC/bert-finetune && cd bert-finetune && cp benchmark_scripts/benchmark_bertbase_hvd_32.sh run_script.sh && bash run_script.sh'" \
--result /results --org nvidian --team sae

ngc batch run --name "Benchmark BERTLARGE HVD 16GB" \
--preempt RUNONCE --total-runtime 100000s --image "nvidian/sae/ai-lab:19.08-tf" --ace nv-us-west-2 --instance dgx1v.16g.8.norm --commandline \
"bash -c 'export N_GPU=8 && git clone --depth 1 https://github.com/NVAITC/bert-finetune && cd bert-finetune && cp benchmark_scripts/benchmark_bertlarge_hvd.sh run_script.sh && bash run_script.sh'" \
--result /results --org nvidian --team sae

ngc batch run --name "Benchmark BERTLARGE HVD 32GB" \
--preempt RUNONCE --total-runtime 100000s --image "nvidian/sae/ai-lab:19.08-tf" --ace nv-us-west-2 --instance dgx1v.32g.8.norm --commandline \
"bash -c 'export N_GPU=8 && git clone --depth 1 https://github.com/NVAITC/bert-finetune && cd bert-finetune && cp benchmark_scripts/benchmark_bertlarge_hvd_32.sh run_script.sh && bash run_script.sh'" \
--result /results --org nvidian --team sae