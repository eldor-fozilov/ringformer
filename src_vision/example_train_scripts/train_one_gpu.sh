#!/bin/bash

model_type=VRingFormer
model_size=B
dataset="imagenet1k"
patch_size=16
extra_info=01

python train/train_acc.py \
    --name ${dataset}_${model_type}_${model_size}_${patch_size}_${extra_info} \
    --dataset $dataset \
    --model_type $model_type \
    --model_size ${model_type}-${model_size}_${patch_size} \
    --train_batch_size 4096 \
    --gradient_accumulation_steps 16 \
    --output_dir checkpoints \
    --num_epochs 50 \
    --early_stop_criterion 10 \
    --warmup_epochs 3 \
    --learning_rate 1e-3 \
    --weight_decay 0.1 \
    --eval_every 1 \

 # --pretrained_dir 