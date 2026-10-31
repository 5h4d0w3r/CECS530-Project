"""Student reference implementations. Inputs/outputs use (batch, sequence, heads, dim).

Scope: equal-length self-attention, equal Q/K/V head counts, no dropout or padding.
Q/K/V projections and the output projection are outside this kernel benchmark.
"""
import math
import torch


def dense_attention(q, k, v, causal=False):
    """Explicit baseline: materialize scores and probabilities. Autograd supported."""
    qh, kh, vh = (x.transpose(1, 2) for x in (q, k, v))
    scores = (qh @ kh.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    if causal:
        n = q.shape[1]
        mask = torch.ones(n, n, device=q.device, dtype=torch.bool).triu(1)
        scores = scores.masked_fill(mask, -torch.inf)
    probabilities = torch.softmax(scores, dim=-1)
    return (probabilities @ vh).transpose(1, 2).contiguous()


def online_attention(q, k, v, causal=False, block_size=32):
    """Teaching implementation of tiled, stable online softmax; not a GPU kernel.

    Keeps a query block's running maximum m, normalizer l and unnormalized
    weighted-value sum acc. Score temporaries are bounded by the tile size;
    choosing a tile smaller than N avoids a full N x N score matrix.
    FP16/BF16 inputs accumulate in FP32; FP64 is preserved for verification.
    Python loops and separate tensor operations make this unsuitable as a
    substitute for FlashAttention's fused kernel.
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    work_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    qh, kh, vh = (x.transpose(1, 2).to(work_dtype) for x in (q, k, v))
    b, h, n, d = qh.shape
    blocks = []
    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        qi = qh[:, :, start:stop]
        m = torch.full((b, h, stop - start, 1), -torch.inf,
                       dtype=work_dtype, device=q.device)
        l = torch.zeros_like(m)
        acc = torch.zeros_like(qi)
        for j in range(0, stop if causal else n, block_size):
            end = min(j + block_size, n)
            scores = (qi @ kh[:, :, j:end].transpose(-2, -1)) / math.sqrt(d)
            if causal:
                rows = torch.arange(start, stop, device=q.device)[:, None]
                cols = torch.arange(j, end, device=q.device)[None, :]
                scores = scores.masked_fill(cols > rows, -torch.inf)
            new_m = torch.maximum(m, scores.amax(dim=-1, keepdim=True))
            alpha = torch.exp(m - new_m)
            p = torch.exp(scores - new_m)
            acc = alpha * acc + p @ vh[:, :, j:end]
            l = alpha * l + p.sum(dim=-1, keepdim=True)
            m = new_m
        blocks.append(acc / l)
    return torch.cat(blocks, dim=2).transpose(1, 2).to(q.dtype).contiguous()
