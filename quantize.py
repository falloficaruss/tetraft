import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def quaternary_quant(w, c, scale):
    t = (1.0 + c) / 2.0
    x = w / scale
    q = torch.where(x < -t, -1.0, torch.zeros_like(x))
    q = torch.where((x >= -t) & (x < 0), -c, q)
    q = torch.where((x >= 0) & (x < t), c, q)
    q = torch.where(x >= t, 1.0, q)
    return q * scale


class QuantizedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, c=0.25):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.lambda_ = 1.0

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        scale = self.weight.abs().max().detach().clamp(min=1e-5)
        w_q = quaternary_quant(self.weight, self.c, scale)
        w = self.weight + self.lambda_ * (w_q - self.weight).detach()
        return F.linear(x, w, self.bias)

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, c={self.c}"
