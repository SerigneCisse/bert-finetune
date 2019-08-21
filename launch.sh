#!/usr/bin/env bash

# for internal use

ngc batch run --name "BERTBASE dbpedia 32GB" \
--preempt RUNONCE --total-runtime 10000s --image "nvidian/sae/ai-lab:19.08-tf" --ace nv-us-west-2 --instance dgx1v.32g.8.norm --commandline \
"bash -c 'export N_GPU=8 && git clone --depth 1 https://github.com/NVAITC/bert-finetune && cd bert-finetune && bash train_dbpedia_bertbase.sh'" \
--result /results --org nvidian --team sae

ngc batch run --name "BERTBASE dbpedia 16GB" \
--preempt RUNONCE --total-runtime 10000s --image "nvidian/sae/ai-lab:19.08-tf" --ace nv-us-west-2 --instance dgx1v.16g.8.norm --commandline \
"bash -c 'export N_GPU=8 && git clone --depth 1 https://github.com/NVAITC/bert-finetune && cd bert-finetune && bash train_dbpedia_bertbase.sh'" \
--result /results --org nvidian --team sae

ngc batch run --name "BERTLARGE dbpedia 32GB" \
--preempt RUNONCE --total-runtime 100000s --image "nvidian/sae/ai-lab:19.08-tf" --ace nv-us-west-2 --instance dgx1v.32g.8.norm --commandline \
"bash -c 'export N_GPU=8 && git clone --depth 1 https://github.com/NVAITC/bert-finetune && cd bert-finetune && bash train_dbpedia_bertlarge.sh'" \
--result /results --org nvidian --team sae

ngc batch run --name "BERTLARGE dbpedia 16GB" \
--preempt RUNONCE --total-runtime 100000s --image "nvidian/sae/ai-lab:19.08-tf" --ace nv-us-west-2 --instance dgx1v.16g.8.norm --commandline \
"bash -c 'export N_GPU=8 && git clone --depth 1 https://github.com/NVAITC/bert-finetune && cd bert-finetune && bash train_dbpedia_bertlarge.sh'" \
--result /results --org nvidian --team sae

