import torch

class LayerNorm(torch.nn.Module):

    def __init__(
            self,
            dim,
            eps: float = 1e-5
    ):
        """LayerNorm
        """
        super().__init__()
        self.dim = dim
        self.eps = eps

        self.scale = torch.nn.Parameter(torch.ones(dim,))
        self.shift = torch.nn.Parameter(torch.zeros(dim,))

    def forward(
            self,
            x: torch.Tensor
    ):
        """
        """
        mu = x.mean(dim=-1)
        sigma2 = x.var(dim=-1, unbiased=False) + self.eps
        sigma = torch.sqrt(sigma2)

        x = (x - mu.unsqueeze(-1)) / sigma.unsqueeze(-1)
        return self.scale * x + self.shift
