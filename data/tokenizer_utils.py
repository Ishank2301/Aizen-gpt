from typing import List, Dict


class Solution:
    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        result = []
        tokens = sorted(vocab.keys(), key=len, reverse=True)

        for number in numbers:
            text = str(number)
            token_list = []
            i = 0

            while i < len(text):
                for token in tokens:
                    if text.startswith(token, i):
                        token_list.append(token)
                        i += len(token)
                        break
                else:
                    i += 1

            result.append(token_list)

        return result

    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        tokens = sorted(vocab.keys(), key=len, reverse=True)

        count = 0
        i = 0

        while i < len(text):
            for token in tokens:
                if text.startswith(token, i):
                    count += 1
                    i += len(token)
                    break
            else:
                i += 1

        return count

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        words = text.split()

        if not words:
            return 0.0

        token_count = self.count_tokens(text, vocab)

        return round(token_count / len(words), 4)