#!/usr/bin/env bash

# for internal use

ngc batch run --name "Experiment 0.01 Dataset" \
--preempt RUNONCE --total-runtime 100000s --image "nvidian/sae/ai-lab:19.08-tf" --ace nv-us-west-2 --instance dgx1v.16g.1.norm --commandline \
"bash -c 'export DATASET_PORTION=0.01 && git clone https://github.com/NVAITC/bert-finetune && cd bert-finetune && git checkout experiments && python3 bert_training.py'" \
--result /results --org nvidian --team sae

ngc batch run --name "Experiment 0.05 Dataset" \
--preempt RUNONCE --total-runtime 100000s --image "nvidian/sae/ai-lab:19.08-tf" --ace nv-us-west-2 --instance dgx1v.16g.1.norm --commandline \
"bash -c 'export DATASET_PORTION=0.05 && git clone https://github.com/NVAITC/bert-finetune && cd bert-finetune && git checkout experiments && python3 bert_training.py'" \
--result /results --org nvidian --team sae

ngc batch run --name "Experiment 0.10 Dataset" \
--preempt RUNONCE --total-runtime 100000s --image "nvidian/sae/ai-lab:19.08-tf" --ace nv-us-west-2 --instance dgx1v.16g.1.norm --commandline \
"bash -c 'export DATASET_PORTION=0.10 && git clone https://github.com/NVAITC/bert-finetune && cd bert-finetune && git checkout experiments && python3 bert_training.py'" \
--result /results --org nvidian --team sae

ngc batch run --name "Experiment 0.20 Dataset" \
--preempt RUNONCE --total-runtime 100000s --image "nvidian/sae/ai-lab:19.08-tf" --ace nv-us-west-2 --instance dgx1v.16g.1.norm --commandline \
"bash -c 'export DATASET_PORTION=0.20 && git clone https://github.com/NVAITC/bert-finetune && cd bert-finetune && git checkout experiments && python3 bert_training.py'" \
--result /results --org nvidian --team sae

ngc batch run --name "Experiment 0.30 Dataset" \
--preempt RUNONCE --total-runtime 100000s --image "nvidian/sae/ai-lab:19.08-tf" --ace nv-us-west-2 --instance dgx1v.16g.1.norm --commandline \
"bash -c 'export DATASET_PORTION=0.30 && git clone https://github.com/NVAITC/bert-finetune && cd bert-finetune && git checkout experiments && python3 bert_training.py'" \
--result /results --org nvidian --team sae

ngc batch run --name "Experiment 0.40 Dataset" \
--preempt RUNONCE --total-runtime 100000s --image "nvidian/sae/ai-lab:19.08-tf" --ace nv-us-west-2 --instance dgx1v.16g.1.norm --commandline \
"bash -c 'export DATASET_PORTION=0.40 && git clone https://github.com/NVAITC/bert-finetune && cd bert-finetune && git checkout experiments && python3 bert_training.py'" \
--result /results --org nvidian --team sae

ngc batch run --name "Experiment 0.50 Dataset" \
--preempt RUNONCE --total-runtime 100000s --image "nvidian/sae/ai-lab:19.08-tf" --ace nv-us-west-2 --instance dgx1v.16g.1.norm --commandline \
"bash -c 'export DATASET_PORTION=0.50 && git clone https://github.com/NVAITC/bert-finetune && cd bert-finetune && git checkout experiments && python3 bert_training.py'" \
--result /results --org nvidian --team sae

ngc batch run --name "Experiment 0.60 Dataset" \
--preempt RUNONCE --total-runtime 100000s --image "nvidian/sae/ai-lab:19.08-tf" --ace nv-us-west-2 --instance dgx1v.16g.1.norm --commandline \
"bash -c 'export DATASET_PORTION=0.60 && git clone https://github.com/NVAITC/bert-finetune && cd bert-finetune && git checkout experiments && python3 bert_training.py'" \
--result /results --org nvidian --team sae

ngc batch run --name "Experiment 0.70 Dataset" \
--preempt RUNONCE --total-runtime 100000s --image "nvidian/sae/ai-lab:19.08-tf" --ace nv-us-west-2 --instance dgx1v.16g.1.norm --commandline \
"bash -c 'export DATASET_PORTION=0.70 && git clone https://github.com/NVAITC/bert-finetune && cd bert-finetune && git checkout experiments && python3 bert_training.py'" \
--result /results --org nvidian --team sae

ngc batch run --name "Experiment 0.80 Dataset" \
--preempt RUNONCE --total-runtime 100000s --image "nvidian/sae/ai-lab:19.08-tf" --ace nv-us-west-2 --instance dgx1v.16g.1.norm --commandline \
"bash -c 'export DATASET_PORTION=0.80 && git clone https://github.com/NVAITC/bert-finetune && cd bert-finetune && git checkout experiments && python3 bert_training.py'" \
--result /results --org nvidian --team sae

ngc batch run --name "Experiment 0.90 Dataset" \
--preempt RUNONCE --total-runtime 100000s --image "nvidian/sae/ai-lab:19.08-tf" --ace nv-us-west-2 --instance dgx1v.16g.1.norm --commandline \
"bash -c 'export DATASET_PORTION=0.90 && git clone https://github.com/NVAITC/bert-finetune && cd bert-finetune && git checkout experiments && python3 bert_training.py'" \
--result /results --org nvidian --team sae

ngc batch run --name "Experiment 1.00 Dataset" \
--preempt RUNONCE --total-runtime 100000s --image "nvidian/sae/ai-lab:19.08-tf" --ace nv-us-west-2 --instance dgx1v.16g.1.norm --commandline \
"bash -c 'export DATASET_PORTION=1.00 && git clone https://github.com/NVAITC/bert-finetune && cd bert-finetune && git checkout experiments && python3 bert_training.py'" \
--result /results --org nvidian --team sae

