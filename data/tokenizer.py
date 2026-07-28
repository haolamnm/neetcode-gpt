from typing import List


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters
        # 2. For each merge step:
        #    a. Count frequency of all adjacent token pairs
        #    b. Find the most frequent pair (break ties lexicographically)
        #    c. Merge all non-overlapping occurrences left to right
        #    d. Record the merge as [token_a, token_b]
        # 3. Return the list of merges performed
        tokens = list(corpus)
        merges = []

        for _ in range(num_merges):
            if len(tokens) < 2:
                break

            pair_counts = defaultdict(int)
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair_counts[pair] += 1

            if not pair_counts:
                break

            # Tie-breaker: maximum count first, then lexicographically smallest pair tuple
            best_pair = max(
                pair_counts.keys(),
                key=lambda p: (pair_counts[p], [-ord(c) for c in "".join(p)])
            )
            # A cleaner Pythonic way to handle (max frequency, min pair lexicographically):
            best_pair = min(
                pair_counts.keys(),
                key=lambda p: (-pair_counts[p], p)
            )

            merges.append(list(best_pair))

            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                    new_tokens.append(best_pair[0] + best_pair[1])
                    i += 2  # Skip the next token as it has been merged
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            tokens = new_tokens

        return merges
