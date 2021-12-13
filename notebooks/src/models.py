from typing import Callable, Dict, Tuple, List, Union

import torch
import torch.nn as nn
import sak
import sak.torch
import sak.torch.nn

class MyModel(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, *args):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.conv = torch.nn.Conv2d(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            padding=self.kernel_size//2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
