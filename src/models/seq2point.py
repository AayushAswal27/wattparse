"""seq2point: 599 samples of mains in, one appliance reading out.

Architecture follows Zhang et al. (2018) - five 1D conv layers then two
dense. `dilations` is the one deviation: with all 1s the receptive field of
the conv stack is only 30 samples (3 minutes at 6s) and the dense layer does
all the long-range work. Widening the dilations lets the convolutions
themselves span most of the window, which is where event duration lives.

    RF = 1 + sum((kernel - 1) * dilation)

    (1,1,1,1,1)    ->  30 samples   published
    (1,2,4,8,16)   -> 140 samples
    (1,4,16,32,64) -> 502 samples   nearly the full window
"""

from __future__ import annotations

import torch
import torch.nn as nn

KERNELS = (10, 8, 6, 5, 5)
FILTERS = (30, 30, 40, 50, 50)
NO_DILATION = (1, 1, 1, 1, 1)


def receptive_field(kernels=KERNELS, dilations=NO_DILATION):
    return 1 + sum((k - 1) * d for k, d in zip(kernels, dilations))


class Seq2Point(nn.Module):
    def __init__(self, window=599, dilations=NO_DILATION, dropout=0.0):
        super().__init__()
        self.window = window
        self.dilations = tuple(dilations)

        layers = []
        c_in = 1
        for c_out, k, d in zip(FILTERS, KERNELS, dilations):
            # Even kernels can't pad symmetrically, so compute it by hand
            # rather than relying on padding="same".
            layers += [nn.Conv1d(c_in, c_out, k, padding=(d * (k - 1)) // 2,
                                 dilation=d),
                       nn.ReLU()]
            c_in = c_out
        self.conv = nn.Sequential(*layers)

        # Output length depends on kernel parity, so measure it instead of
        # assuming it equals the window.
        with torch.no_grad():
            n_flat = self.conv(torch.zeros(1, 1, window)).flatten(1).shape[1]

        head = [nn.Flatten(), nn.Linear(n_flat, 1024), nn.ReLU()]
        if dropout:
            head.append(nn.Dropout(dropout))
        head.append(nn.Linear(1024, 1))
        self.head = nn.Sequential(*head)

    def forward(self, x):
        # x: (batch, window) or (batch, 1, window)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.head(self.conv(x)).squeeze(-1)

    def describe(self):
        n = sum(p.numel() for p in self.parameters())
        conv_n = sum(p.numel() for p in self.conv.parameters())
        return ("Seq2Point window=%d dilations=%s RF=%d params=%s "
                "(conv %s, head %s)"
                % (self.window, self.dilations,
                   receptive_field(KERNELS, self.dilations),
                   f"{n:,}", f"{conv_n:,}", f"{n - conv_n:,}"))