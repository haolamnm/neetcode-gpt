import torch
from typing import List, Tuple

class Solution:
    def batch_loader(self, raw_dataset: str, context_length: int, batch_size: int) -> Tuple[List[List[str]], List[List[str]]]:
        # 1. Tokenize by splitting on whitespace: raw_dataset.split()
        # 2. Generate batch_size random start indices using torch.randint()
        #    Range: [0, len(tokens) - context_length)
        # 3. For each index i, X = tokens[i:i+context_length], Y = tokens[i+1:i+1+context_length]
        torch.manual_seed(0)
        
        tokens = raw_dataset.split()
        max_start = len(tokens) - context_length
        indices = torch.randint(0, max_start, (batch_size,))
        
        X, Y = [], []
        for idx in indices:
            i = idx.item()
            X.append(tokens[i : i + context_length])
            Y.append(tokens[i + 1 : i + 1 + context_length])
            
        return X, Y