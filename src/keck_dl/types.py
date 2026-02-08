from typing import TypeAlias, Union

from tiktoken import Encoding

from keck_dl.llm.tokenizing import BasicTokenizer

TokenizerType: TypeAlias = Union[BasicTokenizer, Encoding]
