#!/bin/bash

model_type=UiT
model_size=B
dataset="imagenet1k"
patch_size=16
extra_info=01_50_epochs

accelerate launch train/train_acc.py \
    --distributed_training True \
    --name ${dataset}_${model_type}_${model_size}_${patch_size}_${extra_info} \
    --dataset $dataset \
    --model_type $model_type \
    --model_size ${model_type}-${model_size}_${patch_size} \
    --train_batch_size 4096 \
    --gradient_accumulation_steps 16 \
    --output_dir checkpoints \
    --num_epochs 50 \
    --early_stop_criterion 10 \
    --warmup_epochs 5 \
    --learning_rate 5e-4 \
    --eval_every 1 \

    # --pretrained_dir 