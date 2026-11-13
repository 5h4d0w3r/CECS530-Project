"""Pipelined decode-attention accelerator: functional model, timing model, scheduler.

Token generation (decode) computes attention for ONE new query against a KV cache
of L earlier tokens. The online-softmax recurrence splits that work into four
stages that each tile of the cache flows through:

    LOAD        K_t, V_t <- T rows of the cache                      (memory)
    SCORE       s = K_t q / sqrt(D)                                  (MAC array)
    SOFTMAX     m' = max(m, max s); a = exp(m - m'); p = exp(s - m'); l' = a*l + sum p
    ACCUMULATE  acc' = a*acc + p V_t                                 (MAC array)
    after the last tile: o = acc / l

Functional model: what each stage computes; checked against dense_attention.
Timing model:     tau_k = ceil(work_k / rate_k) cycles per item in stage k.
Scheduler:        F[m][k] = max(F[m][k-1], F[m-1][k]) + tau[m][k].
"""
import math
from dataclasses import dataclass

import torch

STAGES = ('load', 'score', 'softmax', 'accumulate')


# ---------------------------------------------------------------------------
# Functional model. One sequence: q (H, D); k_cache, v_cache (H, L, D).
# ---------------------------------------------------------------------------

def load(k_cache, v_cache, start, stop):
    """LOAD: rows start..stop-1 of the cache -> K_t, V_t of shape (H, T, D)."""
    return k_cache[:, start:stop], v_cache[:, start:stop]


def score(q, k_tile):
    """SCORE: s_j = q . k_j / sqrt(D) for the T keys of the tile -> (H, T)."""
    return (k_tile @ q.unsqueeze(-1)).squeeze(-1) / math.sqrt(q.shape[-1])


def softmax_update(m, l, s):
    """SOFTMAX: update the running max m and normalizer l; emit alpha and p."""
    m_new = torch.maximum(m, s.amax(dim=-1))
    alpha = torch.exp(m - m_new)
    p = torch.exp(s - m_new.unsqueeze(-1))
    return m_new, alpha * l + p.sum(dim=-1), alpha, p


def accumulate(acc, alpha, p, v_tile):
    """ACCUMULATE: acc' = alpha * acc + sum_j p_j v_j -> (H, D)."""
    return alpha.unsqueeze(-1) * acc + (p.unsqueeze(-2) @ v_tile).squeeze(-2)


def decode_attention_staged(q, k_cache, v_cache, tile=32):
    """One decode step, executed stage by stage, tile by tile. Returns (H, D)."""
    if tile <= 0:
        raise ValueError('tile must be positive')
    heads, length, dim = k_cache.shape
    m = torch.full((heads,), -torch.inf, dtype=q.dtype, device=q.device)
    l = torch.zeros_like(m)
    acc = torch.zeros(heads, dim, dtype=q.dtype, device=q.device)
    for start in range(0, length, tile):
        k_tile, v_tile = load(k_cache, v_cache, start, min(start + tile, length))
        s = score(q, k_tile)
        m, l, alpha, p = softmax_update(m, l, s)
        acc = accumulate(acc, alpha, p, v_tile)
    return acc / l.unsqueeze(-1)


# ---------------------------------------------------------------------------
# Timing model and scheduler. Times are in clock cycles.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Hardware:
    """Work each stage's unit completes per cycle. Defaults are assumptions, not a real chip."""
    clock_hz: float = 1e9
    bytes_per_cycle: int = 64   # memory interface used by LOAD
    macs_per_cycle: int = 64    # SCORE and ACCUMULATE each own a MAC array this wide
    exps_per_cycle: int = 16    # exponential unit used by SOFTMAX


def stage_cycles(tile, head_dim, bytes_per_element, hw=Hardware()):
    """tau_k for one (head, tile) item, in STAGES order."""
    return (math.ceil(2 * tile * head_dim * bytes_per_element / hw.bytes_per_cycle),  # K_t and V_t
            math.ceil(tile * head_dim / hw.macs_per_cycle),                         # T dot products
            math.ceil(tile / hw.exps_per_cycle),                                    # T exponentials
            math.ceil((tile + 1) * head_dim / hw.macs_per_cycle))                   # p V_t and alpha*acc


def decode_items(context, heads, head_dim, tile, bytes_per_element, hw=Hardware(), requests=1):
    """Stage cycles of every item in one decode step: requests x heads x ceil(L/T) items."""
    sizes = [min(tile, context - start) for start in range(0, context, tile)]
    return [stage_cycles(size, head_dim, bytes_per_element, hw)
            for _ in range(requests * heads) for size in sizes]


def simulate(items, pipelined=True):
    """Push items through the stages in order. Returns makespan and per-stage busy cycles.

    Pipelined: stage k starts item m once item m has left stage k-1 AND stage k
    has finished item m-1. Buffers between stages are assumed never to fill.
    Non-pipelined baseline: the same units, but item m enters stage 0 only after
    item m-1 has left the last stage, so no two stages ever work at once.
    """
    if not items:
        raise ValueError('need at least one item')
    previous = [0] * len(items[0])   # F[m-1][k]: when the previous item left each stage
    busy = [0] * len(items[0])
    for tau in items:
        finish = 0 if pipelined else previous[-1]
        current = []
        for k, cycles in enumerate(tau):
            finish = max(finish, previous[k]) + cycles   # F[m][k]
            current.append(finish)
            busy[k] += cycles
        previous = current
    makespan = previous[-1]
    return dict(cycles=makespan, busy=busy, utilization=[b / makespan for b in busy])


def closed_form(tau, n_items):
    """Identical items: T_seq = M*sum(tau); T_pipe = sum(tau) + (M-1)*max(tau)."""
    sequential = n_items * sum(tau)
    pipelined = sum(tau) + (n_items - 1) * max(tau)
    return dict(sequential=sequential, pipelined=pipelined,
                speedup=sequential / pipelined, speedup_limit=sum(tau) / max(tau))


def decode_metrics(context, heads=4, head_dim=64, tile=32, bytes_per_element=2,
                   hw=Hardware(), requests=1):
    """Speedup, utilization, latency and throughput of one attention layer's decode step."""
    items = decode_items(context, heads, head_dim, tile, bytes_per_element, hw, requests)
    sequential, pipelined = simulate(items, pipelined=False), simulate(items)
    seconds = pipelined['cycles'] / hw.clock_hz
    return dict(context=context, items=len(items),
                sequential_cycles=sequential['cycles'], pipelined_cycles=pipelined['cycles'],
                speedup=sequential['cycles'] / pipelined['cycles'],
                utilization=dict(zip(STAGES, pipelined['utilization'])),
                latency_us=seconds * 1e6, tokens_per_s=requests / seconds)


def main():
    print('Pipelined decode-attention simulation')
    print('Assumptions: 4 heads, D=64, tile=32, 2 bytes/element, 1 GHz;')
    print('             64 bytes/cycle, 64 MACs/cycle per compute stage, 16 exps/cycle.')
    print()
    headers = ('Context', 'Schedule', 'Latency/token (us)', 'Throughput (tokens/s)',
               'Utilization: load / score / softmax / accumulate', 'Speedup')
    rows = []
    for context in (128, 512, 2048, 8192):
        items = decode_items(context, heads=4, head_dim=64, tile=32,
                             bytes_per_element=2)
        sequential = simulate(items, pipelined=False)
        pipelined = simulate(items, pipelined=True)
        for label, result in (('Non-pipeline', sequential), ('Pipeline', pipelined)):
            latency_us = result['cycles'] / Hardware().clock_hz * 1e6
            throughput = Hardware().clock_hz / result['cycles']
            utilization = ' / '.join(f'{u * 100:.1f}%'
                                     for u in result['utilization'])
            speedup = sequential['cycles'] / result['cycles']
            rows.append((f'{context:,}', label, f'{latency_us:.3f}',
                         f'{throughput:,.0f}', utilization, f'{speedup:.3f}x'))

    widths = [max(len(headers[i]), *(len(row[i]) for row in rows))
              for i in range(len(headers))]

    def format_row(row):
        return ('| ' + ' | '.join(value.ljust(widths[i])
                                  if i in (1, 4) else value.rjust(widths[i])
                                  for i, value in enumerate(row)) + ' |')

    border = '+-' + '-+-'.join('-' * width for width in widths) + '-+'
    print(border)
    print(format_row(headers))
    print(border)
    for row in rows:
        print(format_row(row))
    print(border)


if __name__ == '__main__':
    main()
