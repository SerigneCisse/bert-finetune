#!/usr/bin/env bash

# scaling

ngc batch run --name "Benchmark BERTBASE 1 32GB" \
--preempt RUNONCE --total-runtime 100000s --image "nvidian/sae/ai-lab:19.08-tf" --ace nv-us-west-2 --instance dgx1v.32g.2.norm --commandline \
"bash -c 'export N_GPU=2 && git clone --depth 1 https://github.com/NVAITC/bert-finetune && cd bert-finetune && cp benchmark_scripts/bertbase_training_32.py run_script.py && python3 run_script.py'" \
--result /results --org nvidian --team sae

ngc batch run --name "Benchmark BERTBASE 1 16GB" \
--preempt RUNONCE --total-runtime 100000s --image "nvidian/sae/ai-lab:19.08-tf" --ace nv-us-west-2 --instance dgx1v.16g.2.norm --commandline \
"bash -c 'export N_GPU=2 && git clone --depth 1 https://github.com/NVAITC/bert-finetune && cd bert-finetune && cp benchmark_scripts/bertbase_training.py run_script.py && python3 run_script.py'" \
--result /results --org nvidian --team sae

ngc batch run --name "Benchmark BERTBASE 2 32GB" \
--preempt RUNONCE --total-runtime 10000s --image "nvidian/sae/ai-lab:19.08-tf" --ace nv-us-west-2 --instance dgx1v.32g.2.norm --commandline \
"bash -c 'export N_GPU=2 && git clone --depth 1 https://github.com/NVAITC/bert-finetune && cd bert-finetune && cp benchmark_scripts/benchmark_bertbase_fastest_32.sh run_script.sh && bash run_script.sh'" \
--result /results --org nvidian --team sae

ngc batch run --name "Benchmark BERTBASE 4 32GB" \
--preempt RUNONCE --total-runtime 10000s --image "nvidian/sae/ai-lab:19.08-tf" --ace nv-us-west-2 --instance dgx1v.32g.4.norm --commandline \
"bash -c 'export N_GPU=4 && git clone --depth 1 https://github.com/NVAITC/bert-finetune && cd bert-finetune && cp benchmark_scripts/benchmark_bertbase_fastest_32.sh run_script.sh && bash run_script.sh'" \
--result /results --org nvidian --team sae

ngc batch run --name "Benchmark BERTBASE 2 16GB" \
--preempt RUNONCE --total-runtime 10000s --image "nvidian/sae/ai-lab:19.08-tf" --ace nv-us-west-2 --instance dgx1v.16g.2.norm --commandline \
"bash -c 'export N_GPU=2 && git clone --depth 1 https://github.com/NVAITC/bert-finetune && cd bert-finetune && cp benchmark_scripts/benchmark_bertbase_fastest.sh run_script.sh && bash run_script.sh'" \
--result /results --org nvidian --team sae

ngc batch run --name "Benchmark BERTBASE 4 16GB" \
--preempt RUNONCE --total-runtime 10000s --image "nvidian/sae/ai-lab:19.08-tf" --ace nv-us-west-2 --instance dgx1v.16g.4.norm --commandline \
"bash -c 'export N_GPU=4 && git clone --depth 1 https://github.com/NVAITC/bert-finetune && cd bert-finetune && cp benchmark_scripts/benchmark_bertbase_fastest.sh run_script.sh && bash run_script.sh'" \
--result /results --org nvidian --team sae
