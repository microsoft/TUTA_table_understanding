#!/bin/bash

python train.py \
    --target formula_prediction \
    --dataset_paths ./data/formula_prediction/enron_train_fortap_input.pt \
    --pretrained_model_path checkpoints/fortap/fortap.bin \
    --output_model_path checkpoints/fortap/fortap-fp.bin \
    --save_checkpoint_steps 50000 \
    --buffer_size 1000 \
    --chunk_size 1000 \
    --batch_size 2 \
    --world_size 1 \
    --gpu 0 \
    --max_seq_len 512