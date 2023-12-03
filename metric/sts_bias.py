import pandas as pd
from collections import defaultdict
# from utils import similarity
from tqdm import tqdm

import torch
import torch.nn as nn
from mlm.wrappers import MLMPredictor

# import ipdb

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# from https://huggingface.co/blog/how-to-train-sentence-transformers
def train_mlm(mlm_model, tokenizer, lm_model_name, args):
    '''
    Train the MLM model using STS-B dataset
    
    Args:
        model: MLM model
        tokenizer: tokenizer
        args: arguments
    
    Returns:
        model: trained MLM model
    '''
    from data.STS.sts_dataset import STSDataset
    from torch.utils.data import DataLoader
    import torch

    # Create predictor MLM
    model = MLMPredictor(mlm_model, mlm_model.config.hidden_size, 1).to(device)

    train_dataset = STSDataset('data/STS/STS-B/sts-train.csv')
    # val_dataset = STSDataset('data/STS/STS-B/sts-dev.csv')

    # Create the DataLoader
    dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)
    # dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # Create the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Create the loss function
    loss_criterion = nn.MSELoss(reduction='mean')

    # We loop over epochs
    for epoch in range(args.epochs):
        model.train()
        print(f"Epoch {epoch+1}/{args.epochs}")
        # We loop over the batches from the DataLoader
        total_loss = 0
        total_count = 0
        for batch in tqdm(dataloader):
            # We set the gradients to zero so that we are ready for the next loop
            optimizer.zero_grad()

            # We extract the input_ids and attention_mask from the batch
            sent1 = batch[0]
            sent2 = batch[1]
            labels = batch[2].to(device)

            encoded_input1 = tokenizer(sent1, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
            encoded_input2 = tokenizer(sent2, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)

            # Concatenate input_ids, attention_mask, and token_type_ids
            # input_ids = torch.cat((encoded_input1['input_ids'], encoded_input2['input_ids'][:, 1:]), dim=1).to(device)
            # attention_mask = torch.cat((encoded_input1['attention_mask'], encoded_input2['attention_mask'][:, 1:]), dim=1).to(device)
            # token_type_ids = torch.cat((encoded_input1['token_type_ids'], encoded_input2['token_type_ids'][:, 1:]), dim=1).to(device)
            
            encoded_input = {'input_ids': torch.cat((encoded_input1['input_ids'], encoded_input2['input_ids'][:, 1:]), dim=1).to(device), 
                            'attention_mask': torch.cat((encoded_input1['attention_mask'], encoded_input2['attention_mask'][:, 1:]), dim=1).to(device), 
                            'token_type_ids': torch.cat((encoded_input1['token_type_ids'], encoded_input2['token_type_ids'][:, 1:]), dim=1).to(device) if 'token_type_ids' in encoded_input1 and 'token_type_ids' in encoded_input2 else None}

            # We feed the input to the model and get the logits
            outputs = model(encoded_input)
            
            #Free memory
            encoded_input = None
            token_type_ids = None
            attention_mask = None
            input_ids = None
            encoded_input1 = None
            encoded_input2 = None
            sent2 = None
            
            
            # We extract the loss from the outputs
            loss = loss_criterion(outputs, labels.float())

            # We use backward to automatically calculate the gradients
            loss.backward()

            # We use the optimizer to update the weights
            optimizer.step()

            total_loss += loss.item()
            total_count += len(sent1)

        # ipdb.set_trace()
        # We print the loss after every epoch
        print(f"Loss after epoch {epoch+1}: {total_loss/total_count}")

        # We save the model after every epoch
        torch.save(
            {
                'epoch': epoch+1, 
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            }, 
                f'metric/models_sts/{lm_model_name}/stsb_epoch_{epoch+1}_debiased_{args.debias_type}.pth' if args.use_debiased else f'metric/models_sts/{lm_model_name}/stsb_epoch_{epoch+1}.pth'
        )
    
    return model
    
def compute_sts_bias(model_config, lm_model_name, args):
    if args.train_model:
        predictor = train_mlm(model_config['model'], model_config['tokenizer'], lm_model_name, args)
    else:
        predictor = MLMPredictor(model_config['model'], model_config['model'].config.hidden_size, 1)
        predictor.load_state_dict(torch.load(f"{args.model_dir}")['model_state_dict'])

    sts_test_bias = pd.read_csv('data/STS/sts-test-bias-final.csv', sep='\t')
    occupation_stats = pd.read_csv('data/STS/STS-B/occupations-stats.tsv', sep='\t')
    n = len(sts_test_bias)/120

    list_of_occ = occupation_stats['occupation'].tolist()
    
    occ = 0
    man_woman = 0
    result = defaultdict(float)
    dif_final = 0

    predictor.eval()
    for sent1, sent2 in tqdm(zip(sts_test_bias['sent1'],sts_test_bias['sent2'])):    
        if model_config['uncased']:
            sent1 = sent1.lower()
            sent2 = sent2.lower()

        # ipdb.set_trace()
        encoded_input1 = model_config['tokenizer'](sent1, return_tensors='pt', padding=True, truncation=True, max_length=128)
        encoded_input2 = model_config['tokenizer'](sent2, return_tensors='pt',padding=True, truncation=True, max_length=128)

        # Concatenate input_ids, attention_mask, and token_type_ids
        # input_ids = torch.cat((encoded_input1['input_ids'], encoded_input2['input_ids'][:,1:]), dim=1)
        # attention_mask = torch.cat((encoded_input1['attention_mask'], encoded_input2['attention_mask'][:,1:]), dim=1)
        # token_type_ids = torch.cat((encoded_input1['token_type_ids'], encoded_input2['token_type_ids'][:,1:]), dim=1)
        # encoded_input = {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids}
        
        encoded_input = {'input_ids': torch.cat((encoded_input1['input_ids'], encoded_input2['input_ids'][:, 1:]), dim=1).to(device), 
                        'attention_mask': torch.cat((encoded_input1['attention_mask'], encoded_input2['attention_mask'][:, 1:]), dim=1).to(device), 
                        'token_type_ids': torch.cat((encoded_input1['token_type_ids'], encoded_input2['token_type_ids'][:, 1:]), dim=1).to(device) if 'token_type_ids' in encoded_input1 and 'token_type_ids' in encoded_input2 else None}

        pred_score = predictor(encoded_input)
        
        if man_woman == 1:
            dif_final = abs(dif_final-pred_score.item())
            result[list_of_occ[occ]] += dif_final/n
            occ+=1
            occ = 0 if occ==60 else occ
            dif_final = 0
            man_woman = 0
        else:
            man_woman+=1
            dif_final = pred_score.item()
    return result