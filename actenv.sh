conda activate accessory
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export PATH=/usr/local/cuda-11.7/bin/${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUDA_HOME=/usr/local/cuda-11.7

<<COMMENT
cd accessory/

# 7B single turn normBias
torchrun --nproc-per-node=1 --master-port 29400 demos/single_turn.py \
--llama_config ../checkpoints/llama2/Llama-2-7b/params.json \
--tokenizer_path ../checkpoints/llama2/Llama-2-7b/tokenizer.model \
--pretrained_path ../checkpoints/llama2/Llama-2-7b/ output/Quantization_Assistant_LLaMA2_Accessory/finetune/sg/alpaca_llamaPeft_normBias_QF_512_7B/epoch3/ \
--quant \
--llama_type llama_peft \

# 13B single turn normBias
torchrun --nproc-per-node=1 --master-port 29500 demos/single_turn.py \
--llama_config ../checkpoints/llama2/Llama-2-13b/params.json \
--tokenizer_path ../checkpoints/llama2/Llama-2-13b/tokenizer.model \
--pretrained_path ../checkpoints/llama2/Llama-2-13b/ output/finetune/sg_legacy/QF_trainableWeight/alpaca_llamaPeft_normBias_QF_512_13B/epoch3/ \
--quant \
--llama_type llama_peft \

# 70B single turn platypus
torchrun --nproc-per-node=1 --master-port 29500 demos/single_turn.py \
--llama_config ../checkpoints/llama2/Llama-2-70b/params.json \
--tokenizer_path ../checkpoints/llama2/Llama-2-70b/tokenizer.model \
--pretrained_path ../checkpoints/llama2/Llama-2-70b/ output/finetune/sg/platypus_normBias_QF_70B/epoch3/ \
--quant --llama_type llama_peft

# 13B multi-modal single run 
torchrun --nproc-per-node=1 --master-port 29501 demos/single_turn_mm.py \
--llama_config ../checkpoints/llama2/Llama-2-13b/params.json configs/model/finetune/sg/llamaPeft_normBiasLora.json \
--tokenizer_path ../checkpoints/llama2/Llama-2-13b/tokenizer.model \
--pretrained_path ../checkpoints/mm/lamaQformerv2_13b/finetuned/ output/Quantization_Assistant_LLaMA2_Accessory/finetune/mm/alpacaLlava_llamaQformerv2Peft_QF_13B_lr1e-4_fixlora/epoch2/ \
--quant \
--llama_type llama_qformerv2_peft \

# 70B multi turn
python demos/multi_turn.py \
--llama_config ../checkpoints/Enderfga/params.json \
--tokenizer_path ../checkpoints/Enderfga/tokenizer.model \
--pretrained_path ../checkpoints/Enderfga/ --quant

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