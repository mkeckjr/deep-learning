import re
from typing import Dict, List

class BasicTokenizer:

    special_tokens = {
        'UNKNOWN': '<|unk|>',
        'ENDOFTEXT': '<|endoftext|>'
    }

    def __init__(
            self, *,
            vocab: Dict[str, int] = None,
            raw_text: str = None
    ):
        if vocab is None and raw_text is None:
            raise RuntimeError(f'Must specify either "vocab" or "raw_text" as a keyword arg')

        if vocab is not None:
            self.set_vocab(vocab)
        else:
            self.set_vocab(BasicTokenizer.create_vocab(raw_text))

    @classmethod
    def create_vocab(
            cls,
            raw_text: str
    ) -> Dict[str, int]:
        raw_words = cls.preprocess(raw_text)
        unique_words = sorted(set(raw_words))
        unique_words.extend(list(cls.special_tokens.values()))

        vocab = {
            word: index
            for index, word in enumerate(unique_words)
        }
        return vocab

    @classmethod
    def preprocess(
            cls,
            raw_text: str
    ):
        words = re.split(r'([,.?!:;"()_\']|--+|\s+)', raw_text)
        words = [word for word in words if word.strip()]
        return words

    def encode(self, raw_text: str) -> List[int]:
        words = self.preprocess(raw_text)
        encoding = [self.vocab[w] if w in self.vocab else self.special_tokens['UNKNOWN']
                    for w in words]
        return encoding

    def set_vocab(self, vocab: Dict[str, int]):
        self.vocab = vocab
        self.inverse = {
            index: word 
            for word, index in vocab.items()
        }

    def decode(self, encoding: List[int]) -> str:
        raw_decode = " ".join([self.inverse[index] for index in encoding])
        decoded = re.sub(r'\s+([.?!:;_"\')])', r'\1', raw_decode)
        return decoded
