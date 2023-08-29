#!/bin/bash

pretrained_path=$1
pretrained_type=meta_ori
llama_config="$2 configs/model/finetune/sg/llamaPeft_normBiasLora.json"
tokenizer_path="$3"
data_config=configs/data/finetune/sg/alpaca.yaml

data_parallel=sdp
model_parallel=1

exp_name=finetune/sg/alpaca_llamaPeft_normBiasLora_r512_original_7B
# exp_name=dummy
echo "exp name: $exp_name"
mkdir -p output/"$exp_name"

torchrun --master_port=$4 --nproc_per_node=8 main_finetune.py \
--output_dir output/"$exp_name" --epochs 4 --warmup_epochs 1 \
--batch_size 2 --accum_iter 4 --num_workers 4 \
--max_words 512 \
--lr 0.0001 --min_lr 0.000005 --clip_grad 2 --weight_decay 0.02 \
--data_parallel "$data_parallel" --model_parallel_size "$model_parallel" --checkpointing \
--llama_type llama_peft --llama_config $llama_config --tokenizer_path "$tokenizer_path" \
--no_visual \
--pretrained_path "$pretrained_path" --pretrained_type="$pretrained_type" \
--data_config $data_config \
2>&1 | tee -a output/"$exp_name"/output.log

echo "exp name: $exp_name"

# --quant --only_save_trainable \
