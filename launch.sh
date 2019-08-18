#!/usr/bin/env bash

# for internal use

ngc batch run --name "Experiment 1% Dataset" \
--preempt RUNONCE --total-runtime 10000s --image "nvidian/sae/ai-lab:19.08-tf" --ace nv-us-west-2 --instance dgx1v.16g.1.norm --commandline \
"bash -c 'export DATASET_PORTION=0.01 && git clone https://github.com/NVAITC/bert-finetune && cd bert-finetune && git checkout experiments && python3 bert_training.py'" \
--result /results --org nvidian --team sae
