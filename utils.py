# from https://github.com/kanekomasahiro/evaluate_bias_in_mlm/blob/main/preprocess.py
import csv
import json
import numpy as np
import pandas as pd

import torch.nn as nn
import torch.nn.functional as F

def read_data_sts_bias(input_file):
    """
    Load data into pandas DataFrame format.
    """
    
    #ipdb.set_trace()

    # from crows_pairs_anonymized.csv we have:
    # sent1 = sent_more
    # sent2 = sent_less
    # direction = stereo_antistereo
    # bias_type = bias_type
    # df_data = pd.DataFrame(columns=['sent1', 'sent2', 'direction', 'bias_type'])

    data = []

    with open(input_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            example = {}
            direction = row['stereo_antistereo']
            example['direction'] = direction
            example['bias_type'] = row['bias_type']

            example['stereotype'] = row['sent_more']
            example['anti-stereotype'] = row['sent_less']
            data.append(example)

    return data

def read_data_cp(input_file):
    """
    Load data into pandas DataFrame format.
    """
    
    #ipdb.set_trace()

    # from crows_pairs_anonymized.csv we have:
    # sent1 = sent_more
    # sent2 = sent_less
    # direction = stereo_antistereo
    # bias_type = bias_type
    # df_data = pd.DataFrame(columns=['sent1', 'sent2', 'direction', 'bias_type'])

    data = []

    with open(input_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            example = {}
            direction = row['stereo_antistereo']
            example['direction'] = direction
            example['bias_type'] = row['bias_type']

            example['stereotype'] = row['sent_more']
            example['anti-stereotype'] = row['sent_less']
            data.append(example)

    return data

def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

def similarity(s1, s2):    
    return np.nan_to_num(cosine(np.nan_to_num(s1), np.nan_to_num(s2)))

def read_data_sts(input_file='data/STS/sts-train-bias-final.csv'):
    return pd.read_csv(input_file)

def read_data_ss(input_file):
    """
    Load data into pandas DataFrame format.
    """
    
    #ipdb.set_trace()

    # from crows_pairs_anonymized.csv we have:
    # sent1 = sent_more
    # sent2 = sent_less
    # direction = stereo_antistereo
    # bias_type = bias_type

    data = []

    with open(input_file, 'r') as f:
        input = json.load(f)
        for annotations in input['data']['intrasentence']:
            example = {}
            example['bias_type'] = annotations['bias_type']
            for annotation in annotations['sentences']:
                example[annotation['gold_label']] = annotation['sentence']
            data.append(example)

    return data

# For Auto-Debiasing from: https://github.com/Irenehere/Auto-Debias/tree/main
def load_word_list(f_path):
    lst = []
    with open(f_path,'r') as f:
        line = f.readline()
        while line:
            lst.append(line.strip())
            line = f.readline()
    return lst

def load_wiki_word_list(f_path):
    vocab = []
    with open(f_path,"r")as f:
        line = f.readline()
        while line:
            vocab.append(line.strip().split()[0])
            line = f.readline()
    return vocab

class JSD(nn.Module):
    def __init__(self,reduction='batchmean'):
        super(JSD, self).__init__()
        self.reduction = reduction

    def forward(self, net_1_logits, net_2_logits):
        net_1_probs = F.softmax(net_1_logits, dim=1)
        net_2_probs= F.softmax(net_2_logits, dim=1)

        total_m = 0.5 * (net_1_probs + net_2_probs)
        loss = 0.0
        loss += F.kl_div(F.log_softmax(net_1_logits, dim=1), total_m, reduction=self.reduction) 
        loss += F.kl_div(F.log_softmax(net_2_logits, dim=1), total_m, reduction=self.reduction) 
     
        return (0.5 * loss) 
    
def clean_vocab(vocab):
    new_vocab = []
    for v in vocab:
        if (v[0] not in ['#','[','.','0','1','2','3','4','5','6','7','8','9']) and len(v)>1:
            new_vocab.append(v)
    return new_vocab


def clean_word_list2(tar1_words_,tar2_words_,tokenizer):
    tar1_words = []
    tar2_words = []
    for i in range(len(tar1_words_)):
        if tokenizer.convert_tokens_to_ids(tar1_words_[i])!=tokenizer.unk_token_id and tokenizer.convert_tokens_to_ids(tar2_words_[i])!=tokenizer.unk_token_id:
            tar1_words.append(tar1_words_[i])
            tar2_words.append(tar2_words_[i])
    return tar1_words, tar2_words

def clean_word_list(vocabs,tokenizer):
    vocab_list = []
    for i in range(len(vocabs)):
        if tokenizer.convert_tokens_to_ids(vocabs[i])!=tokenizer.unk_token_id:
            vocab_list.append(vocabs[i])
    return vocab_list