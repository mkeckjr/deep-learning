from typing import Optional

import numpy
import torch
from torch.nn.parameter import Parameter

class SimpleAttention(torch.nn.Module):

    def __init__(
            self,
            d_in: int,
            d_out: int,
            context_length: int,
            bias: bool = True,
            mask: Optional[str] = None,
            dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.sqrt_d_out = numpy.sqrt(self.d_out)
        self.context_length = context_length

        self.Wq = torch.nn.Linear(d_in, d_out, bias=bias, dtype=dtype)
        self.Wk = torch.nn.Linear(d_in, d_out, bias=bias, dtype=dtype)
        self.Wv = torch.nn.Linear(d_in, d_out, bias=bias, dtype=dtype)

        if mask:
            if mask == 'causal':
                self.register_buffer(
                    'mask',
                    torch.triu(torch.ones(context_length, context_length, dtype=bool), diagonal=1)
                )
        else:
            self.mask = None

    def forward(
            self,
            Q: torch.Tensor,
            K: torch.Tensor,
            V: torch.Tensor
    ) -> torch.Tensor:
        """Perform scaled dot product attention

        Args:
            Q: torch.tensor with (N, T, D) dimensions, where N is the number of sequences,
                T is the number of tokens per sequence, and D is the input dimensionality
            K: torch.tensor with (N, T, D) dimensions, where N is the number of sequences,
                T is the number of tokens per sequence, and D is the input dimensionality
            V: torch.tensor with (N, T, D) dimensions, where N is the number of sequences,
                T is the number of tokens per sequence, and D is the input dimensionality

        Returns:
            torch.Tensor with dimensions (N, T, D), which is the transformed V
        """
        Qin = self.safe(Q)
        Kin = self.safe(K)
        Vin = self.safe(V)

        wq = self.Wq(Qin)
        wk = self.Wk(Kin)
        wv = self.Wv(Vin)

        attn_scores = ((wq @ wk.transpose(1,2)) / self.sqrt_d_out)

        # use the mask
        if self.mask is not None:
            tokens = Qin.shape[1]
            attn_scores.masked_fill_(self.mask[:tokens, :tokens], -torch.inf)

        attn = torch.softmax(attn_scores, dim=-1)
        v_out = attn @ wv

        return v_out

    @classmethod
    def safe(cls, x):
        if len(x.shape) == 2:
            return x.unsqueeze(0)
        return x


class MultiHeadAttention(torch.nn.Module):

    def __init__(
            self,
            d_in: int,
            d_out: int,
            n_heads: int,
            context_length: int,
            bias: bool = True,
            mask: Optional[str] = None,
            dtype: Optional[torch.dtype] = None
    ):
        super().__init__()

        if d_out % n_heads != 0:
            raise ValueError(f'd_out ({d_out}) should be evenly divisible by n_heads ({n_heads})')

        self.d_in = d_in
        self.d_out = d_out
        self.n_heads = n_heads
        self.head_dim = d_out // n_heads
        self.sqrt_d_out = numpy.sqrt(self.head_dim)
        self.context_length = context_length

        self.Wq = torch.nn.Linear(d_in, d_out, bias=bias, dtype=dtype)
        self.Wk = torch.nn.Linear(d_in, d_out, bias=bias, dtype=dtype)
        self.Wv = torch.nn.Linear(d_in, d_out, bias=bias, dtype=dtype)
        self.Wo = torch.nn.Linear(d_out, d_out, bias=bias, dtype=dtype)

        if mask:
            if mask == 'causal':
                self.register_buffer(
                    'mask',
                    torch.triu(torch.ones(context_length, context_length, dtype=bool), diagonal=1)
                )
        else:
            self.mask = None

    def forward(
            self,
            Q: torch.Tensor,
            K: torch.Tensor,
            V: torch.Tensor
    ) -> torch.Tensor:
        """Perform scaled dot product attention

        Args:
            Q: torch.tensor with (N, T, D) dimensions, where N is the number of sequences,
                T is the number of tokens per sequence, and D is the input dimensionality
            K: torch.tensor with (N, T, D) dimensions, where N is the number of sequences,
                T is the number of tokens per sequence, and D is the input dimensionality
            V: torch.tensor with (N, T, D) dimensions, where N is the number of sequences,
                T is the number of tokens per sequence, and D is the input dimensionality

        Returns:
            torch.Tensor with dimensions (N, T, D), which is the transformed V
        """
        Qin = self.safe(Q)
        Kin = self.safe(K)
        Vin = self.safe(V)

        wq = self.Wq(Qin)
        wk = self.Wk(Kin)
        wv = self.Wv(Vin)

        # with multihead, we have to reshape everything
        N, T, dim = wq.shape
        wq = wq.view(N, T, self.n_heads, self.head_dim).transpose(1,2)
        wk = wk.view(N, T, self.n_heads, self.head_dim).transpose(1,2)
        wv = wv.view(N, T, self.n_heads, self.head_dim).transpose(1,2)

        # now each thing is N, num_heads, T, head_dim
        # you can then still just perform matrix multiplication along the last two
        # dimensions, sam as the baseline case
        attn_scores = ((wq @ wk.transpose(2,3)) / self.sqrt_d_out)

        # use the mask if it's there
        if self.mask is not None:
            tokens = Qin.shape[1]
            attn_scores.masked_fill_(self.mask[:tokens, :tokens], -torch.inf)

        attn = torch.softmax(attn_scores, dim=-1)

        # now this attention matrix tensor is shaped
        #     (N, num_heads, T, T)
        # so we can just multiply through by the different Vs and then just
        # reshape again
        v_out = attn @ wv  # v_out: (N, num_heads, T, head_dim)
        v_out.transpose_(1,2)
        v_out = v_out.contiguous().view(N, T, dim)
        v_out = self.Wo(v_out)

        return v_out

    @classmethod
    def safe(cls, x):
        if len(x.shape) == 2:
            return x.unsqueeze(0)
        return x
