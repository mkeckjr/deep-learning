from keck_dl.llm.tokenizing import BasicTokenizer

def test_small_text_readable(small_text_path):
    # find the verdict file if it's here
    assert small_text_path.exists()

    with open(str(small_text_path), 'r', encoding='utf-8') as f:
        text = f.read()

    
def test_basic_tokenizer_preprocess(small_text_path):
    with open(str(small_text_path), 'r', encoding='utf-8') as f:
        text = f.read()

    processed = BasicTokenizer.preprocess(text)
    print(f'Processed is {len(processed)}.')

    tok = BasicTokenizer(raw_text=text)

    print(f'Vocab size is {len(tok.vocab)}')


def test_basic_tokenizer_encode_decode(small_text_path):
    with open(str(small_text_path), 'r', encoding='utf-8') as f:
        text = f.read()

    tok = BasicTokenizer(raw_text=text)

    encoding = tok.encode(text)
    decoding = tok.decode(encoding)
    reencode = tok.encode(decoding)

    assert encoding == reencode
