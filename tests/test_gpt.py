import tiktoken
import torch

from keck_dl.llm.models import GPTBlock, GPTModel, GPT2_DEFAULTS
from keck_dl.llm.generation import generate

def test_gpt_model():
    gpt2_block = GPTBlock(
        GPT2_DEFAULTS['embed_dim'],
        GPT2_DEFAULTS['context_length'],
        GPT2_DEFAULTS['n_heads']
    )

    # construct some random tokens with 2 sequences, 50 tokens
    x = torch.randn((2, 50, GPT2_DEFAULTS['embed_dim']))
    y = gpt2_block(x)

    assert y.shape == x.shape


def test_gpt_model():
    # construct a random integer tensor for this, 2 sequences, 50 tokens
    tokenized = torch.randint(0, GPT2_DEFAULTS['n_vocab']+1, (2, 50))

    gpt2 = GPTModel(
        **GPT2_DEFAULTS
    )

    # print(gpt2)
    param_sizes = [param.numel() for param in gpt2.parameters()]
    total = sum(param_sizes)

    print(f'GPT2 total params: {total}.')

    logits_out = gpt2(tokenized)

    assert logits_out.shape == (2, 50, GPT2_DEFAULTS['n_vocab'])


def test_gpt_gen(small_text_path):
    gpt2 = GPTModel(
        **GPT2_DEFAULTS
    )
    tokenizer = tiktoken.encoding_for_model('gpt2')

    with open(str(small_text_path), 'r', encoding='utf-8') as f:
        text_in = f.read()

    encoded = torch.tensor(tokenizer.encode(text_in), dtype=torch.int32)
    encoded.unsqueeze_(0)
    n_tokens = encoded.shape[1]
    n_in = 100
    generated_tokens = generate(gpt2, encoded[:, :n_in], n=10)

    out_token_list = generated_tokens.squeeze().to(device='cpu', dtype=int).tolist()

    out_text = tokenizer.decode(out_token_list)

    print(out_text)
