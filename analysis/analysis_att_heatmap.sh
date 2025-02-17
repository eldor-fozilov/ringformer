#!/bin/bash

model_type=VRingFormer
model_size=S
patch_size=16
dataset="imagenet100"

python analysis/attn_heatmap.py \
    --name ${dataset}_${model_type}_${model_size}_${patch_size} \
    --model_type ${model_type} \
    --model_size ${model_type}-${model_size}_${patch_size} \
    --dataset $dataset \
    --data_dir datasets/${dataset}/val \
    --output_dir analysis/attn_heatmap \
    --num_samples 10 \
    --pretrained_dir checkpoints/vringformer/imagenet100_VRingFormer_S_16_02_scaled_to_the_size_of_OWF_checkpoint.bin \
