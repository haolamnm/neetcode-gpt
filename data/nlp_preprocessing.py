import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        all_sentences = positive + negative
        unique_words = sorted(set(word for sentence in all_sentences for word in sentence.split()))
        vocab = {word: idx + 1 for idx, word in enumerate(unique_words)}

        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        encoded_tensors = [
            torch.tensor([float(vocab[word]) for word in sentence.split()], dtype=torch.float32)
            for sentence in all_sentences
        ]

        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        return torch.nn.utils.rnn.pad_sequence(encoded_tensors, batch_first=True, padding_value=0.0)