import torch
import torch.nn as nn
import math
from .common import GroupNorm32, zero_module

class QKVAttention(nn.Module):
    """
    A module which performs QKV attention. 
    """
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv, time=None):
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum("bct,bcs->bts", q * scale, k * scale) 
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    """
    def __init__(self, in_channels, n_heads=1, n_head_channels=-1):
        super().__init__()
        self.in_channels = in_channels
        self.norm = GroupNorm32(32, self.in_channels)
        if n_head_channels == -1:
            self.num_heads = n_heads
        else:
            assert in_channels % n_head_channels == 0
            self.num_heads = in_channels // n_head_channels

        self.to_qkv = nn.Conv1d(in_channels, in_channels * 3, 1)
        self.attention = QKVAttention(self.num_heads)
        self.proj_out = zero_module(nn.Conv1d(in_channels, in_channels, 1))

    def forward(self, x, time=None):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.to_qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)