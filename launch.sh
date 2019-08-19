#!/usr/bin/env bash

# for internal use

ngc batch run --name "Experiment 0.0001 Dataset" \
--preempt RUNONCE --total-runtime 100000s --image "nvidian/sae/ai-lab:19.08-tf" --ace nv-us-west-2 --instance dgx1v.16g.1.norm --commandline \
"bash -c 'export DATASET_PORTION=0.0001 && git clone https://github.com/NVAITC/bert-finetune && cd bert-finetune && git checkout experiments && python3 bert_training.py'" \
--result /results --org nvidian --team sae

ngc batch run --name "Experiment 0.0005 Dataset" \
--preempt RUNONCE --total-runtime 100000s --image "nvidian/sae/ai-lab:19.08-tf" --ace nv-us-west-2 --instance dgx1v.16g.1.norm --commandline \
"bash -c 'export DATASET_PORTION=0.0005 && git clone https://github.com/NVAITC/bert-finetune && cd bert-finetune && git checkout experiments && python3 bert_training.py'" \
--result /results --org nvidian --team sae

ngc batch run --name "Experiment 0.001 Dataset" \
--preempt RUNONCE --total-runtime 100000s --image "nvidian/sae/ai-lab:19.08-tf" --ace nv-us-west-2 --instance dgx1v.16g.1.norm --commandline \
"bash -c 'export DATASET_PORTION=0.001 && git clone https://github.com/NVAITC/bert-finetune && cd bert-finetune && git checkout experiments && python3 bert_training.py'" \
--result /results --org nvidian --team sae

ngc batch run --name "Experiment 0.005 Dataset" \
--preempt RUNONCE --total-runtime 100000s --image "nvidian/sae/ai-lab:19.08-tf" --ace nv-us-west-2 --instance dgx1v.16g.1.norm --commandline \
"bash -c 'export DATASET_PORTION=0.005 && git clone https://github.com/NVAITC/bert-finetune && cd bert-finetune && git checkout experiments && python3 bert_training.py'" \
--result /results --org nvidian --team sae

ngc batch run --name "Experiment 0.01 Dataset" \
--preempt RUNONCE --total-runtime 100000s --image "nvidian/sae/ai-lab:19.08-tf" --ace nv-us-west-2 --instance dgx1v.16g.1.norm --commandline \
"bash -c 'export DATASET_PORTION=0.01 && git clone https://github.com/NVAITC/bert-finetune && cd bert-finetune && git checkout experiments && python3 bert_training.py'" \
--result /results --org nvidian --team sae
