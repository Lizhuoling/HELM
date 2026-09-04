import torch
from torch import nn


def swish(value: torch.Tensor) -> torch.Tensor:
    return value * torch.sigmoid(value)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        timesteps = timesteps.float()
        half_dim = self.embedding_dim // 2
        exponent = -torch.arange(
            half_dim,
            dtype=torch.float,
            device=timesteps.device,
        ) * (torch.log(torch.tensor(10000.0, device=timesteps.device)) / half_dim)
        frequencies = timesteps.unsqueeze(-1) * exponent.exp()
        return torch.cat([torch.sin(frequencies), torch.cos(frequencies)], dim=-1)
