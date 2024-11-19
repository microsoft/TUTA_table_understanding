# Table Understanding
This is the official repository of:
+ [TUTA:Tree-based Transformers for Generally Structured Table Pre-training](https://arxiv.org/abs/2010.12537) 
+ [ForTaP:Using Formulae for Numerical-Reasoning-Aware Table Pretraining](https://arxiv.org/abs/2109.07323).

TUTA is a unified pretrained model for understanding generally structured tables. 

Based on TUTA, ForTaP further endows the model with stronger numerical-reasoning skills by pretraining on spreadsheet formulas.


## :beers: News
+ **2024-11-12**: [“Encoding Spreadsheets for Large Language Models”](https://arxiv.org/pdf/2407.09025) at EMNLP 2024.
  
+ **2024-7-15**: [A tutorial on “Large Language Models for Tabular Data”](https://github.com/HaoAreYuDong/Large-Language-Models-for-Tabular-Data/) at SIGIR 2024.
  
+ **2022-7-23**: [A survey on “Table Pretraining: A Survey on Model Architectures, Pretraining Objectives, and Downstream Tasks”](https://arxiv.org/pdf/2201.09745) at IJCAI 2022.

+ **2022-03-22**: We released ForTaP code.

+ **2022-03-08**: ForTaP was accepted by *ACL 2022*. You may find [ForTaP paper](https://arxiv.org/abs/2109.07323) here.

+ **2022-01-09**: We updated cell type classification code for TUTA.

+ **2021-10-29**: We released TUTA code.

+ **2021-9-2**: We released [HiTab](https://github.com/microsoft/HiTab), a large dataset on question answering and data-to-text over complex hierarchical tables. 

+ **2021-8-17**: TUTA was accepted by [*KDD 2021*](https://dl.acm.org/doi/abs/10.1145/3447548.3467434). 

+ **2020-10-21**: We released our [TUTA paper](https://arxiv.org/abs/2010.12537) on arXiv. 


## Code and Usages
Detailed implementations and usages of the pretrain models are shown in their folders:
+ [TUTA](https://github.com/microsoft/TUTA_table_understanding/tree/main/tuta)
+ [ForTaP](https://github.com/microsoft/TUTA_table_understanding/tree/main/fortap)

## Citation
If you find TUTA and ForTaP useful in your research, please consider citing following papers:
```
@inproceedings{wang2021tuta,
  title={TUTA: Tree-based Transformers for Generally Structured Table Pre-training},
  author={Wang, Zhiruo and Dong, Haoyu and Jia, Ran and Li, Jia and Fu, Zhiyi and Han, Shi and Zhang, Dongmei},
  booktitle={Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery \& Data Mining},
  pages={1780--1790},
  year={2021}
}
```

```
@article{cheng2021fortap,
  title={FORTAP: Using Formulae for Numerical-Reasoning-Aware Table Pretraining},
  author={Cheng, Zhoujun and Dong, Haoyu and Cheng, Fan and Jia, Ran and Wu, Pengfei and Han, Shi and Zhang, Dongmei},
  journal={arXiv preprint arXiv:2109.07323},
  year={2021}
}
```

## Contact
If you have any problems regarding the paper or code, please feel free to submit issues in this repository. Or you can reach us by emails.


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
