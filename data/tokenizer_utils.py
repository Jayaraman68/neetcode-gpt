from typing import List, Dict

class Solution:
    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        # Tokenize each number using greedy left-to-right longest match.
        # Return a list of token lists showing how each number gets split.
        result = []
        for number in numbers:
            text = str(number)
            tokens = []
            i = 0
            while i < len(text):
                longest_token = None
                for j in range(i + 1, len(text) + 1):
                    substring = text[i:j]
                    if substring in vocab:
                        longest_token = substring
                if longest_token is not None:
                    tokens.append(longest_token)
                    i += len(longest_token)
                else:
                    # If no match, consume one character
                    tokens.append(text[i])
                    i += 1
            result.append(tokens)
        return result    
        pass

    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        # Count how many tokens the text uses with greedy tokenization.
        # Use greedy left-to-right longest match.
        count = 0
        i = 0
        while i < len(text):
            longest_token = None
            for j in range(i + 1, len(text) + 1):
                substring = text[i:j]
                if substring in vocab:
                    longest_token = substring
            if longest_token is not None:
                count += 1
                i += len(longest_token)
            else:
                count += 1
                i += 1
        return count
        pass

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        # Compute tokens-per-word ratio (fertility).
        # Higher = more expensive and less efficient.
        # Round to 4 decimal places.
        words = text.split(" ")
        word_count = len(words)
        token_count = self.count_tokens(text, vocab)
        fertility = token_count / word_count
        return round(fertility, 4)
        pass
