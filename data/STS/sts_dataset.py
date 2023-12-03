import torch
from torch.utils.data import Dataset
import csv

import ipdb

class STSDataset(Dataset):
    def __init__(self, filename):
        self.data = []
        with open(filename, 'r') as file:
            reader = csv.reader(file, delimiter='\t')
            for row in reader:
                if len(row) < 7:
                    row = row[:5] + row[5].split('\t')
                self.data.append((row[5], row[6], float(row[4])))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]