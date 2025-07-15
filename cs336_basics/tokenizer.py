from typing import Iterable, Iterator

class Tokenizer():
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

    def from_files(self, cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, 'rb') as f:
            pass
        with open(merges_filepath, 'rb') as f:
            pass

        self.special_tokens = special_tokens

    def encode(self, text: str) -> list[int]:
        pass

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        pass

    def decode(self, ids: list[int]) -> str:
        pass

