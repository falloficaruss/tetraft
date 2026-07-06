import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantizeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, scale, c):
        ctx.save_for_backward(weight, scale)
        ctx.c = c
        s = scale.abs()
        x = weight / s
        t = (1.0 + c) / 2.0

        q = torch.where(x < -t, -1.0, torch.zeros_like(x))
        q = torch.where((x >= -t) & (x < 0), -c, q)
        q = torch.where((x >= 0) & (x < t), c, q)
        q = torch.where(x >= t, 1.0, q)

        return q * s

    @staticmethod
    def backward(ctx, grad_output):
        weight, scale = ctx.saved_tensors
        c = ctx.c
        s = scale.abs()
        t = (1.0 + c) / 2.0

        x = weight / s
        sat_mask = (x.abs() <= t).float()

        q = torch.where(x < -t, -1.0, torch.zeros_like(x))
        q = torch.where((x >= -t) & (x < 0), -c, q)
        q = torch.where((x >= 0) & (x < t), c, q)
        q = torch.where(x >= t, 1.0, q)

        grad_weight = grad_output * sat_mask
        grad_scale = (grad_output * q * torch.sign(scale)).sum()
        return grad_weight, grad_scale, None


class QuantizedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, c=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.c = c

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.scale = nn.Parameter(torch.ones(1))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.scale.data.fill_(1.0)
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        q_weight = QuantizeFunction.apply(self.weight, self.scale, self.c)
        return F.linear(x, q_weight, self.bias)

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, c={self.c}"
