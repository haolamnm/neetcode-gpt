from typing import List, Dict

class Solution:
    def _greedy_tokenize(self, text: str, vocab: Dict[str, int]) -> List[str]:
        # Helper method to perform greedy left-to-right longest match tokenization.
        tokens = []
        i = 0
        n = len(text)
        
        while i < n:
            longest_match = None
            longest_length = 0
            
            # Find the longest matching substring in the vocabulary starting at index i
            for token_str in vocab.keys():
                length = len(token_str)
                if text.startswith(token_str, i):
                    if length > longest_length:
                        longest_length = length
                        longest_match = token_str
            
            if longest_match is not None:
                tokens.append(longest_match)
                i += longest_length
            else:
                # If no vocabulary match is found, consume a single character
                tokens.append(text[i])
                i += 1
                
        return tokens

    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        # Tokenize each number (converted to string) using greedy tokenization.
        result = []
        for num in numbers:
            num_str = str(num)
            result.append(self._greedy_tokenize(num_str, vocab))
        return result

    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        # Count the total number of tokens produced by greedy tokenization.
        tokens = self._greedy_tokenize(text, vocab)
        return len(tokens)

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        # Compute tokens-per-word ratio (fertility = token_count / word_count).
        # Rounded to 4 decimal places.
        words = text.split()
        if not words:
            return 0.0
            
        word_count = len(words)
        total_tokens = self.count_tokens(text, vocab)
        
        return round(total_tokens / word_count, 4)