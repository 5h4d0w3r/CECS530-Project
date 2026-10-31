"""Forward-only kernel benchmark; CPU baseline or direct FlashAttention comparison.

Example: python benchmark.py --device cuda --dtype float16 --flash --output gpu.json
"""
import argparse
import gc
import json
import platform
import statistics
import time
from pathlib import Path
import torch
from attention import dense_attention


def measure(fn, device, warmup, repeats):
    for _ in range(warmup):
        fn()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    samples = []
    for _ in range(repeats):
        if device.type == 'cuda':
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            out = fn()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
        else:
            start = time.perf_counter()
            out = fn()
            samples.append((time.perf_counter() - start) * 1000)
        del out
    peak = None
    if device.type == 'cuda':
        gc.collect()
        torch.cuda.synchronize(device)
        initial = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        out = fn()
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device) - initial
        del out
    return dict(median_ms=statistics.median(samples), min_ms=min(samples),
                max_ms=max(samples), samples_ms=samples,
                incremental_peak_allocated_bytes=peak)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--dtype', choices=['float32', 'float16', 'bfloat16'], default='float32')
    p.add_argument('--lengths', nargs='+', type=int, default=[128, 256, 512])
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--dim', type=int, default=64)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--repeats', type=int, default=20)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--flash', action='store_true')
    p.add_argument('--output', default='benchmark_results.json')
    args = p.parse_args()
    if min(args.lengths + [args.batch, args.heads, args.dim, args.repeats, args.threads]) < 1 or args.warmup < 0:
        p.error('sizes/repeats/threads must be positive; warmup must be nonnegative')
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    flash = None
    if args.flash:
        if device.type != 'cuda' or dtype == torch.float32:
            p.error('--flash requires --device cuda and float16 or bfloat16')
        from flash_attn import flash_attn_func
        flash = flash_attn_func
    torch.set_num_threads(args.threads)
    torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32 = False
    report = dict(config=vars(args), torch_version=str(torch.__version__),
                  platform=platform.platform(), cuda_version=torch.version.cuda,
                  hip_version=torch.version.hip, mode='forward, inference, dropout=0',
                  device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.processor(),
                  rows=[])
    if flash:
        import flash_attn
        report['flash_attn_version'] = getattr(flash_attn, '__version__', 'unknown')
    with torch.inference_mode():
        for n in args.lengths:
            q, k, v = [torch.randn(args.batch, n, args.heads, args.dim,
                                   device=device, dtype=dtype) for _ in range(3)]
            for causal in (False, True):
                methods = {'dense': lambda: dense_attention(q, k, v, causal=causal)}
                if flash:
                    methods['flash_attn_func'] = lambda: flash(q, k, v, dropout_p=0., causal=causal)
                # FP32 oracle from the SAME rounded inputs. Not timed.
                oracle = dense_attention(q.float(), k.float(), v.float(), causal=causal)
                for name, fn in methods.items():
                    out = fn().float()
                    max_error = (out - oracle).abs().max().item()
                    relative_l2 = ((out - oracle).norm() / oracle.norm().clamp_min(1e-12)).item()
                    tolerance = 0.03 if dtype == torch.bfloat16 else 0.005 if dtype == torch.float16 else 1e-5
                    torch.testing.assert_close(out, oracle, atol=tolerance, rtol=tolerance)
                    del out
                    row = dict(length=n, causal=causal, method=name,
                               max_abs_error_vs_fp32=max_error, relative_l2_error=relative_l2,
                               score_matrix_bytes=args.batch * args.heads * n * n * q.element_size(),
                               **measure(fn, device, args.warmup, args.repeats))
                    report['rows'].append(row)
                    print(json.dumps(row), flush=True)
                del oracle
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
