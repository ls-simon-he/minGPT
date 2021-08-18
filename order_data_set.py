import logging
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset

class OrderDataset(Dataset):
    
    def __init__(self, data, block_size, split):
        self.data = data #[[112,5564,37598], [112,37598,37598], ..]
        self.block_size = block_size
        self.split = split # train/test
        tokens = set()
        for line in data:
            tokens.update(line)
        vocab = list(tokens)
        self.vocab_size = len(vocab)

        # dict
        self.stoi = { ch:i for i,ch in enumerate(vocab) }
        self.itos = { i:ch for i,ch in enumerate(vocab) }        
        
        num = len(self.data) # total instances
        r = np.random.RandomState(1337) # make deterministic
        perm = r.permutation(num)
        num_test = min(int(num*0.2), 20000) # 20% of the whole dataset, or only up to 1000
        self.ixes = perm[:num_test] if split == 'test' else perm[num_test:]

    def __len__(self):
        return self.ixes.size

    def __getitem__(self, idx):

        idx = self.ixes[idx]
        instance = self.data[idx]
        dix = [self.stoi[s] for s in instance]
        xi = []
        yi = []
        if len(dix) < self.block_size:
            xi = dix[0:-1] + [1] * self.block_size
            yi = dix[1:] + [-100] * self.block_size
        else:
            xi = dix[0:self.block_size]
            yi = dix[1:self.block_size+1] + [-100] * self.block_size
        # x will be input to GPT and y will be the associated expected outputs
        x = torch.tensor(xi[0:self.block_size], dtype=torch.long)
        y = torch.tensor(yi[0:self.block_size], dtype=torch.long) # predict the next token in the sequence
        return x, y
