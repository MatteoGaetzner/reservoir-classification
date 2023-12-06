from argparse import Namespace

import torch
from torch import nn


class HeadMNIST(nn.Module):
    def __init__(self, in_dim: int, device: torch.device = torch.device("mps")) -> None:
        super().__init__()
        self.head = nn.Linear(in_dim, 10, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class RandMatReservClassifierMNIST(nn.Module):
    def __init__(
        self, res_dim: int, device: torch.device = torch.device("mps")
    ) -> None:
        super().__init__()
        self.res_dim = res_dim
        self.res = torch.randn(28 * 28, res_dim, device=device)
        self.head = HeadMNIST(res_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.res @ x)


def load_model(args: Namespace) -> nn.Module:
    if args.model == "rand-mat-res-mnist":
        return RandMatReservClassifierMNIST(args.res_dim, args.device)
    else:
        raise NotImplementedError(f"Unsupported model `{args.model}`.")
