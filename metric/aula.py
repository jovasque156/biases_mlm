# based on https://github.com/kanekomasahiro/evaluate_bias_in_mlm/blob/main/evaluate.py
import torch
import difflib

from collections import defaultdict

import ipdb

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_log_prob_unigram(token_ids,lm):
    """
    Given a sequence of token ids, with one masked token, return the log probability of the masked token.
    """
    # ipdb.set_trace()
    model = lm["model"].to(device)
    log_softmax = lm["log_softmax"]
    # uncased = lm["uncased"]

    # get model hidden states
    # obtain the MaskedLMOutput
    output = model(token_ids)
    # obtain the logits
    logits = output.logits.squeeze(0)
    log_probs = log_softmax(logits)
    token_ids = token_ids.view(-1, 1).detach()
    token_log_probs = log_probs.gather(1, token_ids)[1:-1]
    
    attentions = torch.mean(torch.cat(output.attentions, 0), 0)
    averaged_attentions = torch.mean(attentions, 0)
    averaged_token_attentions = torch.mean(averaged_attentions, 0)
    token_log_probs = token_log_probs.squeeze(1) * averaged_token_attentions[1:-1]

    sentence_log_prob = torch.mean(token_log_probs)
    log_probs = sentence_log_prob

    return log_probs

def compute_aula(data, lm, n=1):
    """
    Score each sentence by masking one word at a time.
    The score for a sentence is the sum of log probability of each word in
    the sentence.
    n = n-gram of token that is masked, if n > 1, we mask tokens with overlapping
    n-grams.
    """
    # ipdb.set_trace()

    tokenizer = lm["tokenizer"]
    mask_token = lm["mask_token"]
    uncased = lm["uncased"]

    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    sent1, sent2 = data["stereotype"], data["anti-stereotype"]

    if uncased:
        sent1 = sent1.lower()
        sent2 = sent2.lower()

    # tokenize
    sent1_token_ids = tokenizer.encode(sent1, return_tensors='pt').to(device)
    sent2_token_ids = tokenizer.encode(sent2, return_tensors='pt').to(device)

    sent1_log_probs = 0.
    sent2_log_probs = 0.

    # compute SSS with w from M masked
    sent1_log_probs = get_log_prob_unigram(sent1_token_ids, lm).item()
    sent2_log_probs = get_log_prob_unigram(sent2_token_ids, lm).item()

    score = {}
    # average over iterations
    score["st"] = sent1_log_probs
    score["at"] = sent2_log_probs

    return score

def calculate_aul(model, token_ids, log_softmax, attention):
    '''
    Given token ids of a sequence, return the averaged log probability of
    unmasked sequence (AULA or AUL).
    '''
    output = model(token_ids).to(device)
    logits = output.logits.squeeze(0)
    log_probs = log_softmax(logits)
    token_ids = token_ids.view(-1, 1).detach()
    token_log_probs = log_probs.gather(1, token_ids)[1:-1]
    if attention:
        attentions = torch.mean(torch.cat(output.attentions, 0), 0)
        averaged_attentions = torch.mean(attentions, 0)
        averaged_token_attentions = torch.mean(averaged_attentions, 0)
        token_log_probs = token_log_probs.squeeze(1) * averaged_token_attentions[1:-1]
    sentence_log_prob = torch.mean(token_log_probs)
    score = sentence_log_prob.item()

    return score