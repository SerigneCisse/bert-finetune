#!/usr/bin/env bash

# TF Performance Features

ngc batch run --name "Benchmark BERTBASE TF 16GB" \
--preempt RUNONCE --total-runtime 100000s --image "nvidian/sae/ai-lab:19.08-tf" --ace nv-us-west-2 --instance dgx1v.16g.8.norm --commandline \
"bash -c 'export N_GPU=8 && git clone --depth 1 https://github.com/NVAITC/bert-finetune && cd bert-finetune && cp benchmark_scripts/benchmark_bertbase_tf.sh run_script.sh && bash run_script.sh'" \
--result /results --org nvidian --team sae

ngc batch run --name "Benchmark BERTLARGE TF 16GB" \
--preempt RUNONCE --total-runtime 100000s --image "nvidian/sae/ai-lab:19.08-tf" --ace nv-us-west-2 --instance dgx1v.16g.8.norm --commandline \
"bash -c 'export N_GPU=8 && git clone --depth 1 https://github.com/NVAITC/bert-finetune && cd bert-finetune && cp benchmark_scripts/benchmark_bertlarge_tf.sh run_script.sh && bash run_script.sh'" \
--result /results --org nvidian --team sae

ngc batch run --name "Benchmark BERTBASE TF 32GB" \
--preempt RUNONCE --total-runtime 100000s --image "nvidian/sae/ai-lab:19.08-tf" --ace nv-us-west-2 --instance dgx1v.32g.8.norm --commandline \
"bash -c 'export N_GPU=8 && git clone --depth 1 https://github.com/NVAITC/bert-finetune && cd bert-finetune && cp benchmark_scripts/benchmark_bertbase_tf_32.sh run_script.sh && bash run_script.sh'" \
--result /results --org nvidian --team sae

ngc batch run --name "Benchmark BERTLARGE TF 32GB" \
--preempt RUNONCE --total-runtime 100000s --image "nvidian/sae/ai-lab:19.08-tf" --ace nv-us-west-2 --instance dgx1v.32g.8.norm --commandline \
"bash -c 'export N_GPU=8 && git clone --depth 1 https://github.com/NVAITC/bert-finetune && cd bert-finetune && cp benchmark_scripts/benchmark_bertlarge_tf_32.sh run_script.sh && bash run_script.sh'" \
--result /results --org nvidian --team sae

