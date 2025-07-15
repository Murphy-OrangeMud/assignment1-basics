from .pretokenization_example import run_pretokenization
from typing import List, Tuple, Dict
from collections import defaultdict


def train_bpe(input_path: str, vocab_size: int, special_tokens: List[str]) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    token_dict = run_pretokenization(input_path, special_tokens)
    print(token_dict)
    vocab = [list(b) for b in token_dict.keys()]
    vocab = set([x.decode('utf-8') for lst in vocab for x in lst])
    merges = []

    def get_stats(token_dict):
        pairs = defaultdict(int)
        for key in token_dict.keys():
            for i in range(len(key) - 1):
                pairs[(key[i].decode('utf-8'), key[i+1].decode('utf-8'))] += token_dict[key]
        return pairs

    def get_max_pair(pairs):
        max_pair = ()
        max_freq = 0
        for key, freq in pairs.items():
            if freq > max_freq:
                max_pair = key
                max_freq = freq
            elif freq == max_freq:
                if key > max_pair:
                    max_pair = key
        return max_pair
        
    def merge(token_dict):
        new_token_dict = defaultdict(int)
        pairs = get_stats(token_dict)
        pair = get_max_pair(pairs)
        replacement = ''.join(pair)
        merges.append((pair[0].encode('utf-8'), pair[1].encode('utf-8')))
        vocab.add(replacement)
        i = 0
        for key, freq in token_dict.items():
            i = 0
            new_word = []
            while i < len(key):
                if i < len(key) - 1 and (key[i].decode('utf-8'), key[i+1].decode('utf-8')) == pair:
                    new_word.append(replacement)
                    i += 2
                else:
                    new_word.append(key[i].decode('utf-8'))
                    i += 1
            new_token_dict[tuple(letter.encode('utf-8') for letter in new_word)] += freq
        
        return new_token_dict

    while len(vocab) < vocab_size:
        new_token_dict = merge(token_dict)
        token_dict = new_token_dict            

    count = 0
    vocab_indexed = defaultdict(bytes)
    for word in vocab:
        vocab_indexed[count] = word.encode('utf-8')
        count += 1
    return vocab_indexed, merges

if __name__ == "__main__":
    input_path = "/home/murphy/Repos/data/TinyStories-valid.txt"
    train_bpe(input_path, 1000, ["<|endoftext|>"])
