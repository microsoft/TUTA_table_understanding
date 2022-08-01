# ForTaP: Using Formulas for Numerical-Reasoning-Aware Table Pretraining
[ForTaP](https://arxiv.org/abs/2109.07323) is a numerical-reasoning-aware tabular encoder pretrained with Excel spreadsheet formulas. 
We design two pretrain tasks to capture the numerical relations in formulas: Numerical Reference Prediction(NRP) and Numerical Calculation Prediction(NCP).
On formula prediction, table question answering, and cell type classification, ForTaP sees large improvements with the enhanced numerical reasoning skills.

## News
+ March 2022: ForTaP code is released.
+ February 2022: ForTaP is accepted by ACL 2022.

## Dependency
The main dependencies of the codebase are python and pytorch-gpu, with no specific version restriction.
You may install other packages(relatively small) as required.

Please feel free to let us know if you have any dependency problems on running!


## Pre-train
We provide the pretrained [ForTaP (fortap.bin)](https://drive.google.com/drive/folders/1pqLyzl-E3Ed-mJncfa_GxTlYLE1NZiRF) checkpoint here.

### From scratch
To pretrain ForTaP from scratch, you may use:
```shell
python tuta/train.py \
    --target fortap \
    --batch_size 8 \
    --pretrained_model_path checkpoints/fortap/tuta.bin \
    --output_model_path checkpoints/fortap/fortap.bin \
    --chunk_size 10000 \
    --buffer_size 400000 \
    --report_steps 200 \
    --world_size 4 \
    --gpu_ranks 0 1 2 3 \
    --dataset_paths ${DATA_SPLIT1}+${DATA_SPLIT2}+.${DATA_SPLIT3}+${DATA_SPLIT4} \
    --repeat_read_dataset \
    --save_checkpoint_steps 50000 \
    --max_seq_len 256 \
    --vocab_path ./vocab/bert_vocab.txt \
    --text_threshold 0.1 \
    --value_threshold 0.05 \
    --mlm_weight 0.3 \
    --sr_weight 1.0 \
    --sr_context_weight 1.0 \
    --nr_weight 1.0 \
    --op_mlm_weight 1.0 \
    --range_mlm_weight 1.0
```
The [tuta](https://drive.google.com/file/d/1pEdrCqHxNjGM4rjpvCxeAUchdJzCYr1g/view) checkpoint for initialization can be downloaded in this link. 
You may save the checkpoints and data under directories `./checkpoints/TASK/` and `./data/TASK/` following the scripts, or otherwise just specify the new path.
The hyper-parameters are exactly the same we use in pretraining. 
Here are some brief explanations: 
+ `chunk_size`: #samples to load at one time
+ `buffer_size`: #samples reached to start pretraining, which enables training before loading the full pretrain corpus
+ `mlm_weight`: BERT MLM task weight, to retain basic MLM skills
+ `sr_weight`: Table-only NRP task weight in paper, i.e., "semantic reference" in code implementation
+ `sr_context_weight`: Table-text NRP task weight in paper
+ `nr_weight`: NCP task weight in paper, i.e., "numerical reasoning" in code implementation
+ `op_mlm_weight`: operator-level Formula MLM weight
+ `range_mlm_weight`: range-level Formula MLM weight

The pretrain corpus for ForTaP is not public available now. All data are processed to the JSON format shown at https://github.com/microsoft/TUTA_table_understanding/tree/main/tuta/data/pretrain, and then are processed to be pt format through prepare.py.


## Fine-tune
### Formula Prediction
For training
```shell
python train.py \
    --target formula_prediction \
    --dataset_paths ./data/formula_prediction/enron_train_fortap_input.pt \
    --pretrained_model_path checkpoints/fortap/fortap.bin \
    --output_model_path checkpoints/formula_prediction/fortap-fp.bin \
    --save_checkpoint_steps 50000 \
    --buffer_size 1000 \
    --chunk_size 1000 \
    --batch_size 2 \
    --world_size 1 \
    --gpu 0 \
    --max_seq_len 512
```

For testing
```shell
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
```

The enron formula prediction dataset [enron_{train/dev/test}.pt](https://drive.google.com/drive/folders/1hPnxXDZe6G_f2sjLJANJz86z-HZ1tWFW?usp=sharing) is provided in the link. 
There are seven items in one sample:
+ `string_matrix`: input table string matrix
+ `(header_rows, header_columns)`: number of header rows/columns. They may be greater than 1 because of hierarchies, which we extract with heuristics.
+ `format_matrix`: input table format matrix, including alignment, indentation, fonts and etc.
+ `table_range`: the excel spreadsheet range of table in the sheet, i.e., "A5:C18"
+ `merged_regions`: record of the merged cells in sheet
+ `formula_dict`: detailed information about the formulas to be predicted in the input table
+ `meta_data`: the path, sheet name, language of original enron file

If you would like to use ForTaP on formula prediction, you may use the pre-tokenized format [enron_{train/test}_fortap_input.pt](https://drive.google.com/drive/folders/1hPnxXDZe6G_f2sjLJANJz86z-HZ1tWFW?usp=sharing)
for ForTaP, otherwise you can use the raw format.

We also provide [the fine-tuned ForTaP on enron (fortap-fp.bin-800000)](https://drive.google.com/drive/folders/1pqLyzl-E3Ed-mJncfa_GxTlYLE1NZiRF) for users to inference formula prediction, 
which is expected to achieve 55.8% top-1 accuracy on test set.


### Question Answering
We evaluate question answer on [HiTab](https://github.com/microsoft/HiTab) fully based on its codebase, 
by replacing the BERT encoder with ForTaP.

To get the best performance, pay attention to some points shown in the paper:
+ We further pretrain ForTaP for 80K steps on table-text NRP setting. 
The [ForTaP further pretrained on table-text NRP (fortap-80000moreNRP.bin)](https://drive.google.com/drive/folders/1pqLyzl-E3Ed-mJncfa_GxTlYLE1NZiRF) checkpoint is also provided in the link. You may use it for the best QA performance.
+ Use a simple attention mask instead of tree attention.
+ Disable warmup steps for parsing.

**Note:** HiTab has corrected about 1.5% samples and upgrades its heuristics to parse table hierarchies 
since ForTaP paper release, thus the current QA performance might be higher than reported.

### Add New Tasks
You can use ForTaP (also TUTA) as a table-text encoder for other downstream tasks by adding another pipeline.
Please reference  the `--target` in `train.py`. 

Specifically, you may modify `train.py`, `trainers.py`, `dynamic_data.py`, `tokenizer.py`,  `model/heads.py`, `model/pretrains.py` to add a new pipeline.




