# from https://github.com/nyu-mll/crows-pairs/blob/master/metric.py
# this is for CPS metric, CrowS-Pairs Score, intrinisc

import argparse
import torch
import json
import csv
from collections import defaultdict

from utils import *
from mlm.wrappers import MaskedModelWrapper, TokenizerWrapper #, ModelWrapper
from metric.cps import compute_cps
from metric.sss import compute_sss
from metric.aula import compute_aula
from metric.sts_bias import compute_sts_bias

# import ipdb
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def evaluate_extrinsic(lm_model, score, args):
    if lm_model == "bert-base-uncased":
        uncased = True
    elif lm_model == 'bert-base-cased':
        uncased = False
    elif lm_model == 'bert-large-uncased':
        uncased = True
    elif lm_model == 'bert-large-cased':
        uncased = False
    elif lm_model == "roberta-base":
        uncased = False
    elif lm_model == "roberta-large":
        uncased = False
    elif lm_model == "albert-base-v2":
        uncased = True

    if args.use_debiased:
        model = MaskedModelWrapper(f'results/{lm_model}/debiased_model_{args.debias_type}', attention=False).model.to(device)
        model.config.output_hidden_states = True
        tokenizer = TokenizerWrapper(lm_model).tokenizer
    else:
        model = MaskedModelWrapper(lm_model, attention=False).model.to(device)
        model.config.output_hidden_states = True
        tokenizer = TokenizerWrapper(lm_model).tokenizer

    lm = {"model": model,
          "tokenizer": tokenizer,
          "uncased": uncased
    }

    if score=='sts-bias':
        results = compute_sts_bias(lm, lm_model, args)

    path = f'{args.output_path}{lm_model}/{score}_{args.debias_type}.csv' if args.use_debiased else  f'{args.output_path}{lm_model}/{score}.csv'
    with open(path, 'w') as f:
        writer = csv.writer(f)
        for key, value in results.items():
            writer.writerow([key, value])

def evaluate_intrinsic(lm_model, score, data, args):
    """
    Evaluate a masked language model using CrowS-Pairs dataset.
    """

    print("Evaluating:", score)
    print("Data:", data)
    print("Model:", lm_model)
    print("-" * 100)
    

    # load data into panda DataFrame
    if data=='cp':
        df_data = read_data_cp('data/CrowS-Pairs/crows_pairs_anonymized.csv' if args.input_file is None else args.input_file)
    if data=='ss':
        df_data = read_data_ss('data/StereoSet/dev.json' if args.input_file is None else args.input_file)

    # supported masked language models
    attention = True if score=='aula'else False

    if args.use_debiased:
        model = MaskedModelWrapper(f'results/{lm_model}/debiased_model_{args.debias_type}', attention=attention).model
        model.config.output_hidden_states = True
        tokenizer = TokenizerWrapper(lm_model).tokenizer
    else:
        model = MaskedModelWrapper(lm_model, attention=attention).model
        model.config.output_hidden_states = True
        tokenizer = TokenizerWrapper(lm_model).tokenizer

    if lm_model=="bert-base-uncased":
        uncased = True
    elif lm_model=='bert-base-cased':
        uncased = False
    elif lm_model=='bert-large-uncased':
        uncased = True
    elif lm_model=='bert-large-cased':
        uncased = False
    elif lm_model=="roberta-base":
        uncased = False
    elif lm_model=="roberta-large":
        uncased = False
    elif lm_model == "albert-base-v2":
        uncased = True

    model.eval()
    if torch.cuda.is_available():
        model.to('cuda')

    # Load mask token
    mask_token = tokenizer.mask_token
    log_softmax = torch.nn.LogSoftmax(dim=0)
    vocab = tokenizer.get_vocab()
    with open(lm_model + ".vocab", "w") as f:
        f.write(json.dumps(vocab))

    lm = {"model": model,
          "tokenizer": tokenizer,
          "mask_token": mask_token,
          "log_softmax": log_softmax,
          "uncased": uncased
    }

    N = 0
    neutral = 0
    # This is used to count the stereo and antistereos
    count = defaultdict(int) 
    scores = defaultdict(int)
    total = len(df_data)
    stereo_score = 0
    results = []
    for input in tqdm(df_data):
        # ipdb.set_trace()
        example = {}
        bias_type = input['bias_type']
        example['bias_type'] = bias_type
        count[bias_type] +=1

        example['stereotype'] = input['stereotype']
        example['anti-stereotype'] = input['anti-stereotype']

        if data=='cp':
            example['direction'] = input['direction']

        if score=='cps':
            score_result = compute_cps(input, lm)
        elif score =='sss':
            lm['log_softmax'] = torch.nn.LogSoftmax(dim=1)
            score_result = compute_sss(input, lm)
        elif score == 'aula':
            lm['log_softmax'] = torch.nn.LogSoftmax(dim=1)
            score_result = compute_aula(input,lm)

        example['score_st'] = score_result['st']
        example['score_at'] = score_result['at']

        N +=1
        if score_result['st'] > score_result['at']:
            stereo_score +=1
            scores[bias_type] +=1
            example['st>at'] = 1
        else:
            example['st>at'] = 0

        results.append(example)

    # ipdb.set_trace()
    bias_score = ((stereo_score)*(100/N))-50
    print('Bias score', bias_score)
    print("=" * 100)
    
    for s in sorted(scores):
        bias_score = (scores[s] * 100/count[s])-50
        print(f'{s}: {bias_score}')

    # ipdb.set_trace()
    path = f'{args.output_path}{lm_model}/{score}_{data}_{args.debias_type}.csv' if args.use_debiased else  f'{args.output_path}{lm_model}/{score}_{data}.csv'
    with open(path, 'w', newline='') as csvfile:
        fieldnames = [f for f in results[0].keys()]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            writer.writerow(result)

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, help="path to input file")
    parser.add_argument("--lm_models", type=str, help="pretrained LM model to use (options: bert-base-uncased, bert-base-cased, bert-large-uncased, bert-large-cased, robert-base, roberta-large, albert-base-v2)")
    parser.add_argument("--output_path", type=str, default='results/', help="path to output file with sentence scores")
    parser.add_argument("--datasets", type=str, default='cp', help="data to analyze [cp, ss, sts-b, nli]")
    parser.add_argument("--scores", type=str, default='cps', help="bias score to compute [cps, sss, aula, sts-bias, nli-bias]")
    parser.add_argument("--bias_type", type=str, default='intrinsic', help="bias type to compute (intrinsic, extrinsic)")
    parser.add_argument("--use_debiased", action='store_true', help='To use debiased mlm')
    parser.add_argument("--debias_type", type=str, help='debias_type, can be race or gender')

    # if nli-bias or sts-bias
    parser.add_argument("--train_model", action='store_true', help="train model on NLI data or STS")
    parser.add_argument("--epochs", type=int, default=5, help="number of epochs to train")
    parser.add_argument("--model_dir", type=str, help="path to use model directory")
    parser.add_argument("--percent", type=float, default=.2, help="The percent of training data to use")
    parser.add_argument("--save_model", type=str, default="models_sts", help="Path to where model to be saved.")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for training.")

    args = parser.parse_args()

    scores = args.scores.split(',') if ',' in args.scores else [args.scores]
    lm_models = args.lm_models.split(',') if ',' in args.lm_models else [args.lm_models]
    datasets = args.datasets.split(',') if ',' in args.datasets else [args.datasets]

    if args.use_debiased and args.debias_type is None:
        raise ValueError('use_debiased is set true but debias_type is not given')

    if args.bias_type=='extrinsic':
        for s in scores:
            if s in ['sts-bias', 'nli-bias', 'biasbios']:
                for m in lm_models:
                        evaluate_extrinsic(m,s,args)
            else:
                raise ValueError(f"Extrinsic bias score {s} is not supported.")
    
    if args.bias_type=='intrinsic':
        for s in scores:
            if s in ['cps', 'sss', 'aula']:
                for m in lm_models:
                    for d in datasets:
                        print(f'Computing {s} for {m} in {d}')
                        evaluate_intrinsic(m,s,d,args)
            else:
                raise ValueError(f"Extrinsic bias score {s} is not supported.")