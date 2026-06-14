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
        for i in range(num_merges):
            pair_counts = {}
            for j in range(len(tokens) - 1):
                pair = (tokens[j], tokens[j + 1])
                if pair in pair_counts:
                    pair_counts[pair] += 1
                else:
                    pair_counts[pair] = 1
            if not pair_counts:
                break
            best_pair = sorted(pair_counts.items(),key=lambda x: (-x[1], x[0]))[0][0]
            token_a, token_b = best_pair
            merges.append([token_a, token_b])
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == token_a and tokens[i + 1] == token_b:
                    new_tokens.append(token_a + token_b)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return merges
        pass
