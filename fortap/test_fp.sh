#!/bin/bash

python train.py \
    --test \
    --target formula_prediction \
    --dataset_paths ./data/formula_prediction/enron_test_fortap_input.pt \
    --pretrained_model_path checkpoints/formula_prediction/fortap-fp.bin-800000 \
    --output_model_path none \
    --buffer_size 5000 \
    --chunk_size 1000 \
    --batch_size 1 \
    --world_size 1 \
    --gpu 0 \
    --max_seq_len 512