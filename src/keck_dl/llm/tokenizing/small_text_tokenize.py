from pathlib import Path

from keck_llm import get_base_directory
from . import BasicTokenizer

if __name__ == '__main__':
    fname = str(Path(get_base_directory() / 'data' / 'small_data.txt'))
    with open(fname, 'r', encoding='utf-8') as f:
        text = f.read()

    tok = BasicTokenizer(raw_text=text)
    encoding = tok.encoding(raw_text)
    decoding = tok.decode(encoding)
