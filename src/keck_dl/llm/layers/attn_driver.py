import torch

from keck_llm.layers import SimpleAttention

if __name__ == '__main__':
    n_dim = 8
    n_tokens = 5
    dtype = torch.float
    x = torch.randn(
        (n_dim, n_tokens),
        dtype=dtype
    )

    attn = SimpleAttention(
        n_dim = n_dim,
        dtype=dtype
    )

    y = attn(x, x, x)
