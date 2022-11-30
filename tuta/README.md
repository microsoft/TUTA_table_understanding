# TUTA

[TUTA](https://arxiv.org/abs/2010.12537) is a unified pretrained model for understanding generally structured tables.
TUTA introduces two mechanisms to utilize structural information: (1) explicit and implicit positional encoding based on bi-tree structure; (2) structure-aware attention to aggregatate neighboring contexts.



## Models
We provide three variants of pre-trained TUTA models: TUTA (-implicit), TUTA-explicit, and TUTA-base.
These pre-trained TUTA variants can be downloaded from:
* [TUTA](https://drive.google.com/file/d/1pEdrCqHxNjGM4rjpvCxeAUchdJzCYr1g/view?usp=sharing)
* [TUTA-explicit](https://drive.google.com/file/d/1FPwn2lQKEf-cGlgFHr4_IkDk_6WThifW/view?usp=sharing)
* [TUTA-base](https://drive.google.com/file/d/1j5qzw3c2UwbVO7TTHKRQmTvRki8vDO0l/view?usp=sharing)


## Training
To run pretraining tasks, simply run
```bash
python train.py                                           \
--dataset_paths="./dataset.pt"                              \
--pretrained_model_path="${tuta_model_dir}/tuta.bin"      \
--output_model_path="${tuta_model_dir}/trained-tuta.bin"

# to enable a quick test, one can run
python train.py  --batch_size 1  --chunk_size 10  --buffer_size 10  --report_steps 1  --total_steps 20

# to enable multi-gpu distributed training, additionally specify 
--world_size 4  --gpu_ranks 0 1 2 3
```
Do make sure that the number of input `dataset_paths` is no less that the `world_size` (i.e. number of `gpu_ranks`). \
One can find more adjustable arguments in the main procedure.


## Downstream tasks

### __Cell Type Classification (CTC)__
To perform the task of cell type classification at downstream: 
- for data processing, use `SheetReader` in the reader.py and `CtcTokenizer` in the tokenizer.py; 
- for fine-tuning, use the `CtcHead` and `TUTA(base)forCTC` in the ./model/ directory.

### __Table Type Classification (TTC)__
To perform the task of table type classification at downstream: 
- for data processing, use `SheetReader` in the reader.py and `TtcTokenizer` in the tokenizer.py; 
- for fine-tuning, use the `TtcHead` and `TUTA(base)forTTC` in the ./model/ directory.

For an end-to-end trial, run:
```bash
python ctc_finetune.py                                           \
--folds_path="${dataset_dir}/folds_deex5.json"                    \
--flat_json_path="${dataset_dir}/deex.jl"                            \
--hier_dir="${dataset_dir}/deex"                            \
--pretrained_model_path="${tuta_model_dir}/tuta.bin"             \
--output_model_path="${tuta_model_dir}/tuta-ctc.bin"              \
--target="tuta"                                                   \
--device_id=0                                                   \
--batch_size=2                                                   \
--max_seq_len=512                                                 \
--max_cell_num=256                                                 \
--epochs_num=40                                                   \
--attention_distance=2                                             
```

A preprocessed dataset of DeEx can be downloaded from:
* [Dataset](https://1drv.ms/u/s!AmjPfaszItzIh0U4pfx-Sxq_wgv4?e=pGF453)
* [Fold](https://1drv.ms/u/s!AmjPfaszItzIh0U4pfx-Sxq_wgv4?e=pGF453)

## Data Pre-processing
For a sample raw table file input, run
```bash
# for SpreadSheet
python prepare.py                          \
--input_dir ./data/pretrain/spreadsheet   \
--source_type sheet                        \
--output_path ./dataset.pt

# for WikiTable
python prepare.py                                      \
--input_path ./data/pretrain/wiki-table-samples.json  \
--source_type wiki                                     \
--output_path ./dataset.pt

# for WDCTable
python prepare.py                         \
--input_dir ./data/pretrain/wdc          \
--source_type wdc                         \
--output_path ./dataset.pt
```

will generate a semi-processed version for pre-training inputs.

Input this data file as an argument into the pre-training script, then the data-loader will dynamically process for three pre-training objectives, namely Masked Language Model (MLM), Cell-Level Cloze(CLC), and Table Context Retrieval (TCR).
