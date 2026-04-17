import torch

def generate(
        model: torch.nn.Module,
        x: torch.Tensor,
        n: int = 1
) -> str:
    context = x  # (N, T) matrix
    for _ in range(n):
        with torch.no_grad():
            logits = model(context[:, -model.context_length:])  # (N, T, |V|)
            logits = logits[:, -1, :]  # (N, |V|)
            probs = torch.softmax(logits, dim=-1)
            next_tokens = torch.argmax(probs, dim=-1)  # (N,)
            context = torch.cat((context, next_tokens.unsqueeze(-1)), dim=-1)

    return context
