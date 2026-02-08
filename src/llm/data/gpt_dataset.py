from typing import List, Tuple

import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

from keck_llm import TokenizerType

class GPTDataset(Dataset):
    def __init__(
            self,
            text: str,
            tokenizer: TokenizerType,
            max_length: int,
            stride: int
    ):
        self.contexts = []
        self.targets = []

        encoding = tokenizer.encode(text)

        for i in range(0, len(encoding) - max_length, stride):
            self.contexts.append(encoding[i:i+max_length])
            self.targets.append(encoding[i+1:i+1+max_length])

    def __len__(self) -> int:
        return len(self.contexts)

    def __getitem__(self, index: int) -> Tuple[List[int], List[int]]:
        return self.contexts[index], self.targets[index]


def gpt_in_memory_dataloader(
        text: str,
        batch_size: int = 4,
        max_length: int = 256,
        stride: int = 128,
        shuffle: bool = True,
        drop_last: bool = True,
        num_workers: int = 0,
        encoding_type: str = 'gpt2'
) -> DataLoader:
    """Create a DataLoader for our simple in-memory GPTDataset

    Args:
        text: string text that will serve as the dataset
        batch_size: loader batch size
        max_length: maximum number of tokens in a context
        stride: stride when loading tokens from dataset; each new batch starts stride
            tokens ahead of the previous batch start token
        shuffle: shuffle batches each time through the dataset if True
        drop_last: don't use the last batch if it is less than batch_size
        num_workers: num additional threads / workers to use when loading data
        encoding_type: the encoding type to get from tiktoken
    """

    tok = tiktoken.get_encoding(encoding_type)
    dataset = GPTDataset(text, tok, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last, num_workers=num_workers
    )

    return dataloader
