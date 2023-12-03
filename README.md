# Biases in MLM
This repository contains the code for the project **Evaluation of Intrinsic and Extrinsic Bias for Debiased Language Model**. Before running and exploring the code of the experiments, please read the instructions below.

## Requirements
First, you should install the packages listed in `requirements.txt`.
```
pip install -r requirements.txt
```

## Datasets
Due to size restrictions, some of the files are not in this repository, such as the embeddings `glove.840B.300d.zip` from Glove project. For a comprehensive list of files, plese review [this Google Drive folder](https://drive.google.com/drive/folders/1ZklPu_5HOUQhKV8eg4eEDkGGgmptB25o?usp=sharing).

## Running experiments
The experiments can be run through two files:

### Computing intrinsic and extrinsic bias
Youy should use the file `main.py` to compute intrinsic and extrinsic measures. The file contains a comprehensive explanation of the arugments. 

For example, if you want to compute CPS, SSS, and AULA intrinsic bias in StereSet and CrawlsPair datasets for RoBERTa model, you can run the below command in your terminal:
```
python3 main.py --lm_models roberta-large --datasets ss,cp --scores cps,sss,aula --bias_type intrinsic
```

### Debiasing MLM using Auto-Debiasing approach
The file `auto_debias.py` should be uses for debiasing Masked Language Models (MLM) using Auto-Debiasing approach.

Each metric and dataset in the arguments must be separated by comma only. If more models are wanted to be used in the experiments, they must be also listed and separated by comma.
