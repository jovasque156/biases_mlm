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
python main.py --lm_models roberta-large --datasets ss,cp --scores cps,sss,aula --bias_type intrinsic
```
Each metric and dataset in the arguments must be separated by comma only. If more models are wanted to be used in the experiments, they must be also listed and separated by comma.

### Debiasing MLM using Auto-Debias approach
The file `auto_debias.py` should be uses for debiasing Masked Language Models (MLM) using Auto-Debias approach. For example, to debias `bert-base-uncased` under `gender` bias type, you should run the following in the terminal:
```
python auto_debias.py --debias_type gender --model_name_or_path bert-base-uncased --prompts_file prompts_bert-base-uncased_gender.txt
```
Note that a name of the `prompts_file` must be given, and it is assumed to be located in `data/debiasing/`. In the example, it is used `prompts_bert-base-uncased_gender.txt`, which contains the cloze-style prompts for bert-base-uncased based on gender biases.

In case of generate cloze-style prompts for another MLM and type of bias, you should use `generate_prompts.py`. For example, for `albert-base-v2` model and race as bias type, you should use the following in your terminal
```
python generate_prompts.py --model_name_or_path albert-base-v2 --debias_typ race
```

In this study we use the cloze-style prompts for `gender` and `race`. For the former, we used `bert-base-uncased`, whereas in the alter we used `albert-base-v2`.
