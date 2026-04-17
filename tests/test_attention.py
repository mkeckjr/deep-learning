import torch

from keck_dl.llm.layers import SimpleAttention, MultiHeadAttention

def test_simple_attention():
    batch_size = 2
    n_dim = 8
    n_tokens = 5
    dtype = torch.float
    x = torch.randn(
        (batch_size, n_tokens, n_dim),
        dtype=dtype
    )

    attn = SimpleAttention(
        n_dim,   # dim_in
        n_dim,   # dim_out
        1024,    # context length
    )

    y = attn(x, x, x)

    assert y.shape == x.shape

    # same with the mask
    attn = SimpleAttention(
        n_dim,   # dim_in
        n_dim,   # dim_out
        1024,    # context length
        mask='causal'
    )

    y = attn(x, x, x)

    assert y.shape == x.shape

    # and let's try that with varying dimensions
    out_dim = 256
    attn = SimpleAttention(
        n_dim,    # dim_in
        out_dim,  # dim_out
        1024,     # context length
        mask='causal'
    )

    y = attn(x, x, x)

    assert y.shape == (batch_size, n_tokens, out_dim)

    # and for cross attention, let's try this; shouldn't make a diff
    z = torch.randn(
        (batch_size, n_tokens, n_dim)
    )

    y = attn(x, z, x)


# now do it for multi-head
def test_multihead_attention():
    batch_size = 2
    n_dim = 64
    n_tokens = 10
    n_heads = 4
    dtype = torch.float
    x = torch.randn(
        (batch_size, n_tokens, n_dim),
        dtype=dtype
    )

    attn = MultiHeadAttention(
        n_dim,    # dim_in
        n_dim,    # dim_out
        n_heads,  # heads
        1024,     # context length
    )

    y = attn(x, x, x)

    assert y.shape == x.shape

    # mask
    attn = MultiHeadAttention(
        n_dim,    # dim_in
        n_dim,    # dim_out
        n_heads,  # heads
        1024,     # context length
        mask='causal'
    )

    y = attn(x, x, x)

    assert y.shape == x.shape

    # vary dims again
    out_dim = 256
    attn = MultiHeadAttention(
        n_dim,    # dim_in
        out_dim,  # dim_out
        n_heads,  # heads
        1024,     # context length
        mask='causal'
    )
    y = attn(x, x, x)

    assert y.shape == (batch_size, n_tokens, out_dim)

    # and same for cross attention
    z = torch.randn(
        (batch_size, n_tokens, n_dim)
    )

    y = attn(x, z, x)


