#!/usr/bin/env bash

# for internal use

# other tests

ngc batch run --name "Benchmark BERTBASE no NVLINK" \
--preempt RUNONCE --total-runtime 10000s --image "nvidian/sae/ai-lab:19.08-tf" --ace nv-us-west-2 --instance dgx1v.32g.8.norm --commandline \
"bash -c 'export N_GPU=8 && git clone --depth 1 https://github.com/NVAITC/bert-finetune && cd bert-finetune && cp benchmark_scripts/benchmark_bertbase_no_nvlink_32.sh run_script.sh && bash run_script.sh'" \
--result /results --org nvidian --team sae

ngc batch run --name "Benchmark BERTLARGE no NVLINK" \
--preempt RUNONCE --total-runtime 10000s --image "nvidian/sae/ai-lab:19.08-tf" --ace nv-us-west-2 --instance dgx1v.32g.8.norm --commandline \
"bash -c 'export N_GPU=8 && git clone --depth 1 https://github.com/NVAITC/bert-finetune && cd bert-finetune && cp benchmark_scripts/benchmark_bertlarge_no_nvlink_32.sh run_script.sh && bash run_script.sh'" \
--result /results --org nvidian --team sae

