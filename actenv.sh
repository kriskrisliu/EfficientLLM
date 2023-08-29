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

# 13B multi-modal fine-tuning
export IMGPATH="../data/coco2017/train2017"; bash exps/finetune/mm/alpacaLlava_llamaQformerv2Peft_QF_13B.sh \
../checkpoints/mm/lamaQformerv2_13b/finetuned/ \
../checkpoints/llama2/Llama-2-13b/params.json \
../checkpoints/llama2/Llama-2-13b/tokenizer.model \


# 7B single turn fine-tuning
bash exps/finetune/sg/alpaca_llamaPeft_normBias_QF.sh \
../checkpoints/llama2/Llama-2-7b/ \
../checkpoints/llama2/Llama-2-7b/params.json \
../checkpoints/llama2/Llama-2-7b/tokenizer.model

COMMENT