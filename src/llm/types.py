from typing import TypeAlias, Union

from tiktoken import Encoding

from .tokenizing import BasicTokenizer

TokenizerType: TypeAlias = Union[BasicTokenizer, Encoding]
