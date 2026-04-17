import torch

from keck_dl.llm.layers import LayerNorm

def test_layernorm():
    batch_size = 2
    n_dim = 8
    n_tokens = 5
    dtype = torch.float
    x = torch.randn(
        (batch_size, n_tokens, n_dim),
        dtype=dtype
    )

    print('LayerNorm')
    ln = LayerNorm(n_dim)
    y = ln(x)

    print(x)
    print('--------')
    print(y)


