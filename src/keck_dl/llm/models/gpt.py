import tiktoken
import torch
from typing import Optional

from keck_dl.llm.layers import LayerNorm, MultiHeadAttention

GPT2_DEFAULTS = {
    'n_vocab': 50257,
    'embed_dim': 768,
    'context_length': 1024,
    'n_heads': 12,
    'n_blocks': 12,
    'dropout': 0.1,
    'mask': 'causal',
    'bias': False,
    'dtype': torch.float,
}

class GPTFFN(torch.nn.Module):
    def __init__(
            self,
            embed_dim: int,
            
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(self.embed_dim, 4*self.embed_dim),
            torch.nn.GELU(),
            torch.nn.Linear(4*self.embed_dim, self.embed_dim)
        )

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        return self.layers(x)


class GPTBlock(torch.nn.Module):
    def __init__(
            self,
            embed_dim: int,
            context_length: int,
            n_heads: int,
            dropout: float = 0.1,
            bias: bool = False,
            dtype: Optional[torch.dtype] = None,
            mask: Optional[str] = None
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.context_length = context_length
        self.n_heads = n_heads
        self.dropout = dropout

        self.ln1 = LayerNorm(embed_dim)
        self.ln2 = LayerNorm(embed_dim)
        self.drop1 = torch.nn.Dropout(self.dropout)
        self.drop2 = torch.nn.Dropout(self.dropout)

        self.attn = MultiHeadAttention(
            d_in=embed_dim,
            d_out=embed_dim,
            n_heads=n_heads,
            context_length=context_length,
            bias=bias,
            mask=mask,
            dtype=dtype
        )
        self.ffn = GPTFFN(embed_dim)

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        residual = x
        x = self.ln1(x)
        x = self.attn(x, x, x)
        x = self.drop1(x)
        x = x + residual

        residual = x
        x = self.ln2(x)
        x = self.ffn(x)
        x = self.drop2(x)
        x = x + residual

        return x

class GPTModel(torch.nn.Module):
    def __init__(
            self,
            n_vocab: int,
            embed_dim: int,
            context_length: int,
            n_heads: int,
            n_blocks: int,
            bias: bool = False,
            dropout: float = 0.1,
            mask: Optional[str] = None,
            dtype: Optional[str] = None,
    ):
        super().__init__()
        self.n_vocab = n_vocab
        self.embed_dim = embed_dim
        self.context_length = context_length

        self.embeddings = torch.nn.Embedding(n_vocab, embed_dim)
        self.pos_encoding = torch.nn.Embedding(context_length, embed_dim)
        self.drop_in = torch.nn.Dropout(dropout)

        self.blocks = torch.nn.Sequential(*[
            GPTBlock(
                embed_dim,
                context_length,
                n_heads,
                bias=bias,
                dropout=dropout,
                mask=mask,
                dtype=dtype
            )
            for _ in range(n_blocks)
        ])

        self.ln_out = LayerNorm(embed_dim)
        self.Wout = torch.nn.Linear(embed_dim, n_vocab, bias=False)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of a GPT-style Transformer

        Args:
            x: (N, T) matrix of tokenized inputs
        """
        n_tokens = x.shape[-1]
        tokens = self.embeddings(x)
        tokens = tokens + self.pos_encoding(torch.arange(n_tokens, device=x.device))
        tokens = self.drop_in(tokens)

        features = self.blocks(tokens)

        # final layer stuff here
        out = self.ln_out(features)
        out = self.Wout(out)

        return out
