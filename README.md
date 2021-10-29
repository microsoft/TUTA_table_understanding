# Table Understanding with Tree-based Attention (TUTA)

Please keep tuned after we complete the internal process of publishing TUTA's model and code. 
Welcome to contact us for more technique details and discussions: zhiruow@andrew.cmu.edu, hadong@microsoft.com

## :beers: Updates

+ **Stay tuned!**: Code and data of cell type classification.

+ **2021-10-29**: Code of TUTA.

+ **2021-9-2**: We released [HiTab](https://github.com/microsoft/HiTab), a large dataset on question answering and data-to-text over complex hierarchical tables. 

+ **2021-8-17**: We presented our work in [KDD'21](https://dl.acm.org/doi/abs/10.1145/3447548.3467434). 

+ **2020-10-21**: We released our [paper](https://arxiv.org/abs/2010.12537) on arXiv. 

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
--dataset_paths="../dataset.pt"                              \
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


## Data Pre-processing
For a sample raw table file input, run
```bash
# for SpreadSheet
python prepare.py                          \
--input_dir ../data/pretrain/spreadsheet   \
--source_type sheet                        \
--output_path ../dataset.pt

# for WikiTable
python prepare.py                                      \
--input_path ../data/pretrain/wiki-table-samples.json  \
--source_type wiki                                     \
--output_path ../dataset.pt

# for WDCTable
python prepare.py                         \
--input_dir ../data/pretrain/wdc          \
--source_type wdc                         \
--output_path ../dataset.pt
```

will generate a semi-processed version for pre-training inputs.

Input this data file as an argument into the pre-training script, then the data-loader will dynamically process for three pre-training objectives, namely Masked Language Model (MLM), Cell-Level Cloze(CLC), and Table Context Retrieval (TCR).
## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
