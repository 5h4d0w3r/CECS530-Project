"""CPU experiments for online attention, backward, projections and training.

Each configuration runs in its own subprocess. Timings exclude interpreter
startup. Peak RSS includes interpreter/libraries, inputs, model, warmups and
all repetitions: it is total process peak, not incremental tensor allocation.
"""
import argparse
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path


def worker(config):
    import torch
    from attention import dense_attention, online_attention
    from models import SelfAttentionLayer, TinyTransformer

    torch.set_num_threads(config['threads'])
    torch.set_num_interop_threads(1)
    torch.manual_seed(17)
    b, n, h, d = (config[k] for k in ('batch', 'length', 'heads', 'dim'))
    causal, method = config['causal'], config['method']
    phase, scope = config['phase'], config['scope']
    model = None
    if scope == 'kernel':
        inputs = [torch.randn(b, n, h, d, requires_grad=True) for _ in range(3)]
        if method == 'dense':
            forward = lambda: dense_attention(*inputs, causal=causal)
        else:
            forward = lambda: online_attention(*inputs, causal=causal,
                                                block_size=config['block_size'])
    else:
        inputs = [torch.randn(b, n, h * d, requires_grad=True)]
        cls = SelfAttentionLayer if scope == 'layer' else TinyTransformer
        model = cls(h, d, method, config['block_size'])
        forward = lambda: model(inputs[0], causal=causal)
    grad_shape = (b, n, h, d) if scope == 'kernel' else (b, n, h * d)
    upstream = torch.randn(grad_shape)
    optimizer = None
    target = None
    if phase == 'train_step':
        # Model training updates parameters; synthetic features are fixed.
        inputs[0].requires_grad_(False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        target = torch.randn(grad_shape)

    def zero_grad():
        for x in inputs:
            x.grad = None
        if model is not None:
            model.zero_grad(set_to_none=True)

    def prepare():
        if phase != 'train_step':
            zero_grad()
        # A fresh graph is prepared OUTSIDE the backward-only timing interval.
        return forward() if phase == 'backward' else None

    def operation(prepared):
        if phase == 'forward':
            with torch.inference_mode():
                return forward()
        if phase == 'backward':
            prepared.backward(upstream)
        elif phase == 'forward_backward':
            forward().backward(upstream)
        elif phase == 'train_step':
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(forward(), target)
            loss.backward()
            optimizer.step()
            return loss.detach()

    initial_loss = None
    if target is not None:
        with torch.inference_mode():
            initial_loss = torch.nn.functional.mse_loss(forward(), target).item()
    samples = []
    for i in range(config['warmup'] + config['repeats']):
        prepared = prepare()
        start = time.perf_counter_ns()
        result = operation(prepared)
        elapsed = (time.perf_counter_ns() - start) / 1e6
        if i >= config['warmup']:
            samples.append(elapsed)
        if result is not None and not torch.isfinite(result).all():
            raise ValueError('nonfinite output or loss')
        del prepared, result
    final_loss = None
    if target is not None:
        with torch.inference_mode():
            final_loss = torch.nn.functional.mse_loss(forward(), target).item()
    if phase != 'forward':
        gradients = [x.grad for x in inputs if x.requires_grad]
        if model is not None:
            gradients += [p.grad for p in model.parameters()]
        if not gradients or any(g is None or not torch.isfinite(g).all() for g in gradients):
            raise ValueError('missing or nonfinite gradients')
    # macOS reports bytes; Linux reports KiB.
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != 'darwin':
        peak *= 1024
    return dict(**config, median_ms=statistics.median(samples),
                min_ms=min(samples), max_ms=max(samples), samples_ms=samples,
                process_peak_rss_bytes=peak, initial_loss=initial_loss,
                final_loss=final_loss, finite_checks_passed=True,
                torch_version=str(torch.__version__), python_version=platform.python_version())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker-json', help=argparse.SUPPRESS)
    p.add_argument('--lengths', nargs='+', type=int, default=[128, 256, 512])
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--dim', type=int, default=64)
    p.add_argument('--block-size', type=int, default=32)
    p.add_argument('--scopes', nargs='+', choices=['kernel', 'layer', 'transformer'],
                   default=['kernel', 'layer', 'transformer'])
    p.add_argument('--phases', nargs='+', choices=['forward', 'backward', 'forward_backward', 'train_step'],
                   default=['forward', 'backward', 'forward_backward', 'train_step'])
    p.add_argument('--methods', nargs='+', choices=['dense', 'online'], default=['dense', 'online'])
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--repeats', type=int, default=20)
    p.add_argument('--output', default='extended_results.json')
    args = p.parse_args()
    if args.worker_json:
        print(json.dumps(worker(json.loads(args.worker_json))))
        return
    if min(args.lengths + [args.batch, args.heads, args.dim, args.block_size,
                          args.threads, args.repeats]) < 1 or args.warmup < 0:
        p.error('sizes/repeats/threads must be positive; warmup must be nonnegative')
    report = dict(platform=platform.platform(), config=vars(args), dtype='float32',
                  device='cpu', seed=17, flash_attention_executed=False,
                  memory_definition='isolated process lifetime peak RSS, including libraries and inputs; not incremental tensor allocation',
                  backward_definition='forward graph preparation and gradient clearing excluded from backward-only timing',
                  training_definition='one pre-norm Transformer block, fixed random features/targets, MSE, SGD lr=0.01; zero_grad + forward + loss + backward + optimizer.step',
                  rows=[], failures=[])
    environment = dict(os.environ, OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads))
    output = Path(args.output)
    for n in args.lengths:
        for causal in (False, True):
            for scope in args.scopes:
                phases = ('train_step',) if scope == 'transformer' else ('forward', 'backward', 'forward_backward')
                for phase in phases:
                    if phase not in args.phases:
                        continue
                    for method in args.methods:
                        config = dict(length=n, batch=args.batch, heads=args.heads,
                                      dim=args.dim, block_size=args.block_size,
                                      warmup=args.warmup, repeats=args.repeats,
                                      threads=args.threads, causal=causal,
                                      scope=scope, phase=phase, method=method)
                        completed = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                                    '--worker-json', json.dumps(config)],
                                                   capture_output=True, text=True, env=environment)
                        if completed.returncode:
                            report['failures'].append(dict(config=config, stderr=completed.stderr))
                            print('FAILED ' + json.dumps(config), flush=True)
                        else:
                            row = json.loads(completed.stdout)
                            report['rows'].append(row)
                            print(f"N={n} causal={causal} {scope}/{phase}/{method}: {row['median_ms']:.3f} ms, RSS {row['process_peak_rss_bytes']/2**20:.1f} MiB", flush=True)
                        # Save after every case so interrupted runs retain evidence.
                        output.write_text(json.dumps(report, indent=2) + '\n')
    if report['failures']:
        raise SystemExit('Some cases failed; see saved failures')


if __name__ == '__main__':
    main()
