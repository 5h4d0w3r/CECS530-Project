"""Small models that expose the cost of projections and a full training step."""
import torch
from torch import nn
from attention import dense_attention, online_attention


class SelfAttentionLayer(nn.Module):
    def __init__(self, heads=4, dim=64, method='dense', block_size=32):
        super().__init__()
        self.heads, self.dim = heads, dim
        self.method, self.block_size = method, block_size
        width = heads * dim
        self.qkv = nn.Linear(width, 3 * width)
        self.out = nn.Linear(width, width)

    def forward(self, x, causal=False):
        b, n, _ = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, self.dim).unbind(2)
        if self.method == 'dense':
            out = dense_attention(q, k, v, causal)
        elif self.method == 'online':
            out = online_attention(q, k, v, causal, self.block_size)
        else:
            raise ValueError('unknown attention method: ' + self.method)
        return self.out(out.reshape(b, n, self.heads * self.dim))


class TinyTransformer(nn.Module):
    """One pre-norm Transformer block; no dropout, token embedding or vocabulary projection.

    Fixed synthetic feature/target tensors keep this a compute experiment,
    rather than a text quality experiment.
    """
    def __init__(self, heads=4, dim=64, method='dense', block_size=32):
        super().__init__()
        width = heads * dim
        self.norm1 = nn.LayerNorm(width)
        self.attention = SelfAttentionLayer(heads, dim, method, block_size)
        self.norm2 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(),
                                 nn.Linear(4 * width, width))

    def forward(self, x, causal=False):
        x = x + self.attention(self.norm1(x), causal)
        return x + self.ffn(self.norm2(x))
