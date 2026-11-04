"""Token generation with a KV cache: the non-pipelined decode baseline.

An autoregressive model produces text one token at a time. Prefill runs the whole prompt
in one causal pass and stores every token's keys and values (the KV cache). Each
later decode step feeds only the newest token: its k, v are appended to the cache
and its single query attends to all cached keys.

TinyLM = token + position embedding -> pre-norm Transformer blocks -> LayerNorm ->
vocabulary projection, decoded greedily. Weights are random: this measures computation, not
text quality. Attention is the dense reference from attention.py.
"""
import time
import torch
from torch import nn
from attention import dense_attention
from models import TinyTransformer


class KVCache:
    """Preallocated keys/values for every layer, stored (B, H, max_len, D).

    Each head's cache is one contiguous (max_len, D) block, so attention reads it
    without copying. Only the first `length` positions hold valid data.
    """
    def __init__(self, layers, batch, max_len, heads, dim, dtype=torch.float32, device=None):
        shape = (batch, heads, max_len, dim)
        self.keys = [torch.empty(shape, dtype=dtype, device=device) for _ in range(layers)]
        self.values = [torch.empty(shape, dtype=dtype, device=device) for _ in range(layers)]
        self.max_len, self.length = max_len, 0

    def update(self, layer, k, v):
        """Write n new k, v (B, n, H, D) after the valid ones; return all as (B, L, H, D) views."""
        stop = self.length + k.shape[1]
        if stop > self.max_len:
            raise ValueError('KV cache is full')
        self.keys[layer][:, :, self.length:stop] = k.transpose(1, 2)
        self.values[layer][:, :, self.length:stop] = v.transpose(1, 2)
        return (self.keys[layer][:, :, :stop].transpose(1, 2),
                self.values[layer][:, :, :stop].transpose(1, 2))

    def advance(self, n):
        self.length += n

    def nbytes(self, tokens=None):
        """Bytes of K and V for `tokens` positions (default: the valid length)."""
        batch, heads, _, dim = self.keys[0].shape
        tokens = self.length if tokens is None else tokens
        return 2 * len(self.keys) * batch * heads * tokens * dim * self.keys[0].element_size()


def cached_attention(layer, x, cache, index):
    """SelfAttentionLayer.forward, except keys/values go into and come from the cache."""
    b, n, _ = x.shape
    if n > 1 and cache.length:
        raise ValueError('only the first (prefill) step may contain several tokens')
    q, k, v = layer.qkv(x).reshape(b, n, 3, layer.heads, layer.dim).unbind(2)
    keys, values = cache.update(index, k, v)
    # Prefill: n queries over the same n keys, causal. Decode: one query over all
    # cached keys, every one of which is in the past, so no mask is needed.
    out = dense_attention(q, keys, values, causal=n > 1)
    return layer.out(out.reshape(b, n, layer.heads * layer.dim))


class TinyLM(nn.Module):
    def __init__(self, vocab=4096, layers=4, heads=4, dim=64, max_positions=4096):
        super().__init__()
        width = heads * dim
        self.heads, self.dim = heads, dim
        self.tokens = nn.Embedding(vocab, width)
        self.positions = nn.Embedding(max_positions, width)
        self.blocks = nn.ModuleList(TinyTransformer(heads, dim, 'dense') for _ in range(layers))
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab, bias=False)

    def new_cache(self, batch, max_len):
        weight = self.tokens.weight
        return KVCache(len(self.blocks), batch, max_len, self.heads, self.dim,
                       weight.dtype, weight.device)

    def forward(self, tokens, cache=None, last_only=False):
        """tokens (B, n) -> logits (B, n, vocab), or (B, 1, vocab) when last_only.

        Without a cache: one causal pass over the n tokens.
        With a cache: the n tokens follow the cached ones and are appended to it.
        """
        start = cache.length if cache is not None else 0
        n = tokens.shape[1]
        x = self.tokens(tokens) + self.positions(torch.arange(start, start + n, device=tokens.device))
        for index, block in enumerate(self.blocks):
            if cache is None:
                x = block(x, causal=True)
            else:   # TinyTransformer.forward with cached attention
                x = x + cached_attention(block.attention, block.norm1(x), cache, index)
                x = x + block.ffn(block.norm2(x))
        if cache is not None:
            cache.advance(n)
        if last_only:
            x = x[:, -1:]
        return self.head(self.norm(x))


def synchronize(device):
    """Wait for queued GPU work so wall-clock timings are complete. No-op on CPU."""
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps':
        torch.mps.synchronize()


@torch.inference_mode()
def generate(model, prompt, new_tokens, use_cache=True):
    """Greedy decoding. Returns (new tokens (B, new_tokens), seconds per step).

    seconds[0] is prefill: whole prompt -> first new token (time to first token).
    seconds[i], i >= 1, is one decode step: one more token for every sequence.
    use_cache=False re-runs the entire sequence every step (the naive approach).
    Cache allocation happens before the first timed step.
    """
    cache = model.new_cache(prompt.shape[0], prompt.shape[1] + new_tokens) if use_cache else None
    sequence, step_input, generated, seconds = prompt, prompt, [], []
    for _ in range(new_tokens):
        synchronize(prompt.device)
        start = time.perf_counter()
        logits = model(step_input if use_cache else sequence, cache, last_only=True)
        token = logits[:, -1].argmax(dim=-1, keepdim=True)
        synchronize(prompt.device)
        seconds.append(time.perf_counter() - start)
        generated.append(token)
        sequence = torch.cat([sequence, token], dim=1)
        step_input = token
    return torch.cat(generated, dim=1), seconds
