import torch

from keck_llm.models import GPTBlock, GPTModel, GPT2_DEFAULTS

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
