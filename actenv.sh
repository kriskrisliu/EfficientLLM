conda activate accessory
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export PATH=/usr/local/cuda-11.7/bin/${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUDA_HOME=/usr/local/cuda-11.7

<<COMMENT
cd accessory/

# 7B single turn normBiasLora
torchrun --nproc-per-node=1 --master-port 29400 demos/single_turn.py \
--llama_type llama_peft \
--llama_config ../checkpoints/llama2/Llama-2-7b/params.json configs/model/finetune/sg/llamaPeft_normBiasLora.json \
--tokenizer_path ../checkpoints/llama2/Llama-2-7b/tokenizer.model \
--pretrained_path ../checkpoints/effiLLaMA/ \


-------------------------------
# TRAINING

# 7B single turn fine-tuning
bash exps/finetune/sg/alpaca_llamaPeft_normBiasLora_Effi.sh \
../checkpoints/llama2/Llama-2-7b/ \
../checkpoints/llama2/Llama-2-7b/params.json \
../checkpoints/llama2/Llama-2-7b/tokenizer.model \
1122

COMMENT