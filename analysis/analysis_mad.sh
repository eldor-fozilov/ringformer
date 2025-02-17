#!/bin/bash

model_type=ViT
model_size=S
patch_size=16
dataset="imagenet100"
extra_info=original

python mean_attn_dist.py \
    --name ${dataset}_${model_type}_${model_size}_${patch_size} \
    --model_type ${model_type} \
    --model_size ${model_type}-${model_size}_${patch_size} \
    --dataset $dataset \
    --data_dir data/${dataset}/val \
    --output_dir mean_attn_dist \
    --num_samples 500 \
    --extra_info ${extra_info} \
    --pretrained_dir checkpoints/vit/imagenet100_ViT_S_16_01_checkpoint.bin \