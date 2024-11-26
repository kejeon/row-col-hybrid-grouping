#!/bin/bash
{
    curr_dir="$( cd "$(dirname "$0")" ; pwd -P )"
    # Change it when necessary
    # source activate pt

    train_id="exp_cifar10_resnet20q_124832_recursive"
    train_stamp="$(date +"%Y%m%d_%H%M%S")"
    train_id=${train_id}_${train_stamp}

    result_dir="$curr_dir/results/$train_id"
    mkdir -p $result_dir

    python -u train_cifar10.py \
        --model resnet20q \
        --dataset cifar10 \
        --train_split train \
        --lr 0.01 \
        --lr_decay "10,15,20,25" \
        --epochs 30 \
        --optimizer adam \
        --weight-decay 0.0 \
        --results-dir $result_dir \
        --pretrain './pretrained/resnet20q_any_hard.pth.tar' \
        --weight_bit_width_list "2,4,16,256,0"
} && exit
