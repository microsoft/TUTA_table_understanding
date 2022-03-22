#!/bin/bash

python tuta/train.py \
    --batch_size 8 \
    --pretrained_model_path checkpoints/fortap/tuta.bin \
    --output_model_path checkpoints/formula_prediction/fortap.bin \
    --chunk_size 10000 \
    --buffer_size 400000 \
    --report_steps 200 \
    --world_size 4 \
    --gpu_ranks 0 1 2 3 \
    --dataset_paths ${DATA_SPLIT1}+${DATA_SPLIT2}+.${DATA_SPLIT3}+${DATA_SPLIT4} \
    --repeat_read_dataset \
    --save_checkpoint_steps 50000 \
    --target fortap \
    --max_seq_len 256 \
    --vocab_path ./vocab/bert_vocab.txt \
    --text_threshold 0.1 \
    --value_threshold 0.05 \
    --mlm_weight 0.3 \
    --sr_weight 1.0 \
    --nr_weight 1.0 \
    --op_mlm_weight 1.0 \
    --range_mlm_weight 1.0 \
    --sr_context_weight 1.0