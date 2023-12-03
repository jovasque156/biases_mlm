#based on https://github.com/moinnadeem/StereoSet/blob/master/code/models/models.py

import torch.nn as nn
from transformers import BertTokenizer, BertForMaskedLM, BertModel
from transformers import AlbertTokenizer, AlbertForMaskedLM, AlbertModel
from transformers import RobertaTokenizer, RobertaForMaskedLM, RobertaModel

# import ipdb

class MLMPredictor(nn.Module):
    def __init__(self, mlm, hidden_size, outpout_size=1):
        super(MLMPredictor, self).__init__()
        self.mlm = mlm
        self.out = nn.Linear(hidden_size, outpout_size)

    def forward(self, encoded_inputs):
        # ipdb.set_trace()
        outputs = self.mlm(**encoded_inputs, output_hidden_states=True)
        # pooled_output = outputs[1]
        # Let's use the CLS token embedding of the last layer in MLM
        cls_hidden_state = outputs.hidden_states[-1][:,0,:]
        score = self.out(cls_hidden_state)
        return score

class ModelWrapper:
    def __init__(self, model_name_or_path):
        if 'roberta' in model_name_or_path:
            self.model = RobertaModel.from_pretrained(model_name_or_path)
        elif 'albert' in model_name_or_path:
            self.model = AlbertModel.from_pretrained(model_name_or_path)
        elif 'bert' in model_name_or_path:
            self.model = BertModel.from_pretrained(model_name_or_path)
        else:
            raise ValueError(f"Unsupported model name or path: {model_name_or_path}")


class MaskedModelWrapper:
    def __init__(self, model_name_or_path, attention):
        if 'roberta' in model_name_or_path:
            self.model = RobertaForMaskedLM.from_pretrained(model_name_or_path, output_attentions=attention)
        elif 'albert' in model_name_or_path:
            self.model = AlbertForMaskedLM.from_pretrained(model_name_or_path, output_attentions=attention)
        elif 'bert' in model_name_or_path:
            self.model = BertForMaskedLM.from_pretrained(model_name_or_path, output_attentions=attention)
        else:
            raise ValueError(f"Unsupported model name or path: {model_name_or_path}")
        
class TokenizerWrapper:
    def __init__(self, tokenizer_name_or_path):
        if 'roberta' in tokenizer_name_or_path:
            self.tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name_or_path)
        elif 'albert' in tokenizer_name_or_path:
            self.tokenizer = AlbertTokenizer.from_pretrained(tokenizer_name_or_path)
        elif 'bert' in tokenizer_name_or_path:
            self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name_or_path)
        else:
            raise ValueError(f"Unsupported tokenizer name or path: {tokenizer_name_or_path}")

# import transformers
# from torch import nn

# class BertLM(transformers.BertPreTrainedModel):
#     def __init__(self):
#         pass

#     def __new__(self, pretrained_model):
#         return transformers.BertForMaskedLM.from_pretrained(pretrained_model)

# class BertNextSentence(transformers.BertPreTrainedModel):
#     def __init__(self, pretrained_model):
#         pass

#     def __new__(self, pretrained_model):
#         return transformers.BertForNextSentencePrediction.from_pretrained(pretrained_model)

# class RoBERTaLM(transformers.BertPreTrainedModel):
#     def __init__(self, pretrained_model):
#         pass

#     def __new__(self, pretrained_model):
#         return transformers.RobertaForMaskedLM.from_pretrained(pretrained_model)

# class XLNetLM(transformers.BertPreTrainedModel):
#     def __init__(self, pretrained_model):
#         pass

#     def __new__(self, pretrained_model):
#         return transformers.XLNetLMHeadModel.from_pretrained(pretrained_model)

# class XLMLM(transformers.BertPreTrainedModel):
#     def __init__(self, pretrained_model):
#         pass

#     def __new__(self, pretrained_model):
#         return transformers.XLMWithLMHeadModel.from_pretrained(pretrained_model)

# class GPT2LM(transformers.GPT2PreTrainedModel):
#     def __init__(self, pretrained_model):
#         pass

#     def __new__(self, pretrained_model):
#         return transformers.GPT2LMHeadModel.from_pretrained(pretrained_model)

# class ModelNSP(nn.Module):
#     def __init__(self, pretrained_model, nsp_dim=300):
#         super(ModelNSP, self).__init__()
#         self.pretrained2model = {"xlnet": "XLNetModel", "bert": "BertModel", "roberta": "RobertaModel", "gpt2": "GPT2Model"}
#         self.model_class = self.pretrained2model[pretrained_model.lower().split("-")[0]]
        
#         # Below is similar to transformers.model_class.from_pretrained(pretrained_model.lower()), 
#         # but by using getattr(), we can handle the case the attribute doesn't exist
#         self.core_model = getattr(transformers, self.model_class).from_pretrained(pretrained_model)
#         self.core_model.train()
        
#         # if pretrained_model=="gpt2-xl":
#           # for name, param in self.core_model.named_parameters():
#             # print(name)
#             # # freeze word token embeddings and word piece embeddings!
#             # if 'wte' in name or 'wpe' in name: 
#               # param.requires_grad = False
#         hidden_size = self.core_model.config.hidden_size
#         self.nsp_head = nn.Sequential(nn.Linear(hidden_size, nsp_dim), 
#             nn.Linear(nsp_dim, nsp_dim),
#             nn.Linear(nsp_dim, 2))
#         self.criterion = nn.CrossEntropyLoss()

#     def forward(self, input_ids, token_type_ids=None, attention_mask=None, next_sentence_label=None, \
#             position_ids=None, head_mask=None, labels=None):

#         if 'Roberta' in self.model_class or 'GPT2' in self.model_class:
#             outputs = self.core_model(input_ids, attention_mask=attention_mask)#, token_type_ids=token_type_ids)
#         else:
#             outputs = self.core_model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

#         # assert len(outputs)==2

#         if 'gpt2' in self.model_class.lower():
#             output = outputs[0].mean(dim=1)
#             logits = self.nsp_head(output)
#         elif 'XLNet' in self.model_class: 
#             logits = self.nsp_head(outputs[0][:,0,:]) 
#         else:
#             logits = self.nsp_head(outputs[1]) 

#         if labels is not None:
#             output = logits
#             if type(output)==tuple:
#                 output = output[0]

#             loss = self.criterion(logits, labels)
#             return output, loss
#         return logits 