"""Token-generation baseline: prefill, then greedy decoding with a KV cache.

For each prompt length (context) and batch size it records:
  time to first token  prefill of the whole prompt, producing the first new token
  decode latency       one decode step = one new token for every sequence in the batch
  decode throughput    batch * decode steps / total decode time (tokens/s)
  no-cache step        one token produced by re-running the whole context (reference)
  KV cache bytes       2 * layers * batch * heads * cached tokens * head_dim * bytes

The default model (16 heads x 64 = width 1024, 4 layers, ~240 MB of FP32 weights)
is deliberately larger than a CPU's on-chip caches. Large token-generation model weights live in main
memory, so each decode step must stream them; a model that fits in cache would be
limited by compute instead and would misrepresent decode. Dense prefill of the
whole batch materializes B*H*N^2 scores, which bounds the default sweep on 8 GB.

Example: python benchmark_decode.py --contexts 128 512 1024 --batches 1 4 8
"""
import argparse
import json
import platform
import statistics
from pathlib import Path
import torch
from generation import TinyLM, generate


def p90(samples):
    return statistics.quantiles(samples, n=10)[-1] if len(samples) > 1 else samples[0]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--contexts', nargs='+', type=int, default=[128, 512, 1024])
    p.add_argument('--batches', nargs='+', type=int, default=[1, 4, 8])
    p.add_argument('--new-tokens', type=int, default=32)
    p.add_argument('--layers', type=int, default=4)
    p.add_argument('--heads', type=int, default=16)
    p.add_argument('--dim', type=int, default=64)
    p.add_argument('--vocab', type=int, default=4096)
    p.add_argument('--warmup', type=int, default=1)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--no-cache-repeats', type=int, default=1,
                   help='samples of the (slow) no-cache reference step per configuration')
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--device', default='cpu')
    p.add_argument('--dtype', choices=['float32', 'float16', 'bfloat16'], default='float32')
    p.add_argument('--output', default='decode_results.json')
    args = p.parse_args()
    sizes = args.contexts + args.batches + [args.layers, args.heads, args.dim, args.vocab,
                                            args.repeats, args.no_cache_repeats, args.threads]
    if min(sizes) < 1 or args.warmup < 0:
        p.error('sizes/repeats/threads must be positive; warmup must be nonnegative')
    if args.new_tokens < 2:
        p.error('--new-tokens must be at least 2 so that decode steps are timed')

    torch.set_num_threads(args.threads)
    device, dtype = torch.device(args.device), getattr(torch, args.dtype)
    torch.manual_seed(17)
    model = TinyLM(args.vocab, args.layers, args.heads, args.dim,
                   max_positions=max(args.contexts) + args.new_tokens).to(device, dtype)
    element = next(model.parameters()).element_size()
    kv_bytes_per_token = 2 * args.layers * args.heads * args.dim * element   # one sequence
    report = dict(config=vars(args), torch_version=str(torch.__version__),
                  python_version=platform.python_version(), platform=platform.platform(),
                  processor=platform.processor(), device=str(device), attention='dense',
                  decoding='greedy, random weights, KV cache preallocated outside timing',
                  weight_bytes=sum(t.numel() * t.element_size() for t in model.parameters()),
                  kv_bytes_per_token=kv_bytes_per_token, rows=[])
    output = Path(args.output)
    print(f"weights {report['weight_bytes'] / 2**20:.1f} MiB, "
          f"KV cache {kv_bytes_per_token / 2**10:.1f} KiB per token per sequence", flush=True)
    for context in args.contexts:
        for batch in args.batches:
            prompt = torch.randint(args.vocab, (batch, context), device=device)
            for _ in range(args.warmup):
                generate(model, prompt, args.new_tokens)
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            prefill_ms, decode_runs_ms = [], []
            for _ in range(args.repeats):
                _, seconds = generate(model, prompt, args.new_tokens)
                prefill_ms.append(seconds[0] * 1e3)
                decode_runs_ms.append([s * 1e3 for s in seconds[1:]])
            peak = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None
            # Without a cache, every token costs a full forward pass over the context.
            no_cache_ms = [generate(model, prompt, 1, use_cache=False)[1][0] * 1e3
                           for _ in range(args.no_cache_repeats)]
            decode = [t for run in decode_runs_ms for t in run]
            row = dict(context=context, batch=batch, new_tokens=args.new_tokens,
                       time_to_first_token_ms=statistics.median(prefill_ms),
                       decode_ms_median=statistics.median(decode), decode_ms_p90=p90(decode),
                       decode_tokens_per_s=batch * len(decode) / (sum(decode) / 1e3),
                       no_cache_step_ms_median=statistics.median(no_cache_ms),
                       kv_cache_bytes=kv_bytes_per_token * batch * (context + args.new_tokens - 1),
                       peak_gpu_allocated_bytes=peak,
                       prefill_ms_samples=prefill_ms, decode_ms_samples=decode_runs_ms,
                       no_cache_ms_samples=no_cache_ms)
            report['rows'].append(row)
            output.write_text(json.dumps(report, indent=2) + '\n')   # keep partial results
            print(f"context={context:5d} batch={batch:3d}  "
                  f"TTFT {row['time_to_first_token_ms']:9.2f} ms  "
                  f"decode {row['decode_ms_median']:7.3f} ms/step (p90 {row['decode_ms_p90']:.3f})  "
                  f"{row['decode_tokens_per_s']:8.0f} tok/s  "
                  f"no-cache step {row['no_cache_step_ms_median']:9.2f} ms  "
                  f"KV {row['kv_cache_bytes'] / 2**20:7.2f} MiB", flush=True)


if __name__ == '__main__':
    main()
