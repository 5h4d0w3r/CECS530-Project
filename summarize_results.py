"""Create CSV, figures and a repository report from completed experiments."""
import csv
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent / '.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent


def read(name):
    report = json.loads((ROOT / name).read_text())
    if report['failures']:
        raise ValueError('Failed experiment: ' + name)
    return report['rows']


def select(rows, **filters):
    return sorted([r for r in rows if all(r[k] == v for k, v in filters.items())],
                  key=lambda r: r['length'])


def save(fig, name):
    fig.savefig(ROOT / (name + '.png'), dpi=160, bbox_inches='tight')
    svg_path = ROOT / (name + '.svg')
    fig.savefig(svg_path, bbox_inches='tight')
    svg_path.write_text('\n'.join(line.rstrip() for line in svg_path.read_text().splitlines()) + '\n')
    plt.close(fig)


def main():
    rows = read('extended_results.json')
    metadata = json.loads((ROOT / 'extended_results.json').read_text())
    hardware = json.loads((ROOT / 'hardware.json').read_text())
    memory = read('memory_results.json')
    tiles = [r for r in rows if r['length'] == 512 and r['scope'] == 'kernel'
             and r['method'] == 'online' and r['phase'] in ('forward', 'forward_backward')]
    for tile in (16, 64, 128):
        tiles += read(f'tile_{tile}_results.json')
    additional = [r for r in tiles if r['block_size'] != 32]
    fields = ['scope', 'phase', 'method', 'length', 'causal', 'block_size',
              'median_ms', 'min_ms', 'max_ms', 'process_peak_rss_bytes',
              'initial_loss', 'final_loss']
    with (ROOT / 'results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore', lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows + memory + additional)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False})
    panels = [('kernel', 'forward'), ('kernel', 'backward'),
              ('kernel', 'forward_backward'), ('layer', 'forward_backward'),
              ('layer', 'forward'), ('layer', 'backward'),
              ('transformer', 'train_step')]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    for ax, (scope, phase) in zip(axes.flat, panels):
        for method, color in (('dense', '#1565c0'), ('online', '#c44e00')):
            for causal, style in ((False, '-'), (True, '--')):
                series = select(rows, scope=scope, phase=phase, method=method, causal=causal)
                ax.plot([r['length'] for r in series], [r['median_ms'] for r in series],
                        marker='o', color=color, linestyle=style,
                        label=method + (' causal' if causal else ' noncausal'))
        ax.set(title=f'{scope}: {phase}', xlabel='Sequence length', ylabel='Median latency (ms)')
        ax.set_yscale('log')
        ax.grid(alpha=.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].axis('off')
    axes.flat[-1].legend(handles, labels, loc='center', frameon=False)
    fig.suptitle('CPU FP32 · B=1, H=4, D=64 · one thread · tile=32 · logarithmic latency axes')
    save(fig, 'latency')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for method, color in (('dense', '#1565c0'), ('online', '#c44e00')):
        for causal, style in ((False, '-'), (True, '--')):
            series = select(memory, method=method, causal=causal)
            label = method + (' causal' if causal else ' noncausal')
            axes[0].plot([r['length'] for r in series], [r['process_peak_rss_bytes']/2**20 for r in series],
                         marker='o', color=color, linestyle=style, label=label)
            axes[1].plot([r['length'] for r in series], [r['median_ms'] for r in series],
                         marker='o', color=color, linestyle=style, label=label)
    axes[0].set(ylabel='Total process peak RSS (MiB)', title='Isolated-process memory; includes libraries')
    axes[1].set(ylabel='Median latency (ms)', title='Forward inference at longer sequences')
    for ax in axes:
        ax.set_xlabel('Sequence length')
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    save(fig, 'memory')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, phase in zip(axes, ('forward', 'forward_backward')):
        for causal, color in ((False, '#1565c0'), (True, '#c44e00')):
            series = sorted(select(tiles, phase=phase, causal=causal), key=lambda r: r['block_size'])
            ax.plot([r['block_size'] for r in series], [r['median_ms'] for r in series],
                    marker='o', color=color, label='causal' if causal else 'noncausal')
        ax.set(title=phase, xlabel='Tile size (tokens)', ylabel='Median latency (ms)')
        ax.set_xscale('log', base=2)
        ax.set_xticks([16, 32, 64, 128], labels=['16', '32', '64', '128'])
        ax.grid(alpha=.2)
        ax.legend()
    fig.suptitle('Educational online attention · N=512 · CPU FP32 · one thread')
    save(fig, 'tiles')

    first = rows[0]
    lines = ['# Extended benchmark results', '',
             f"Measured on {hardware['cpu']}, {hardware['ram_bytes']/2**30:.1f} GiB RAM; {metadata['platform']}.",
             f"Python {first['python_version']}, PyTorch {first['torch_version']}, CPU FP32, {first['threads']} CPU thread(s), seed 17.",
             f"B={first['batch']}, H={first['heads']}, D={first['dim']}; tile={first['block_size']} unless noted. Both masking modes were measured.", '',
             'All six correctness tests passed; see [test output](test_results.txt).',
             'All experiment rows passed finite-output/gradient checks. FlashAttention did not execute.', '',
             '## Latency at N=512', '',
             'Median milliseconds; five warmups and twenty measured repetitions.', '',
             '| Workload | Mask | Dense | Online | Online / dense |',
             '| --- | --- | ---: | ---: | ---: |']
    for scope, phase in panels:
        for causal in (False, True):
            dense = select(rows, scope=scope, phase=phase, causal=causal, method='dense', length=512)[0]
            online = select(rows, scope=scope, phase=phase, causal=causal, method='online', length=512)[0]
            lines.append(f"| {scope}/{phase} | {'causal' if causal else 'noncausal'} | {dense['median_ms']:.3f} | {online['median_ms']:.3f} | {online['median_ms']/dense['median_ms']:.2f}x |")
    lines += ['', 'The ratio is latency, so values above 1 mean online was slower.',
              'Backward-only excludes building the fresh forward graph and clearing gradients.',
              'Forward-plus-backward includes graph construction, but excludes clearing gradients.',
              'Transformer train_step includes zero_grad, forward, MSE, backward and SGD.', '',
              '![Latency comparisons](latency.png)', '', '## Larger forward-only memory cases', '',
              'One warmup and five repetitions. Peak RSS includes the complete isolated process;',
              'it is not incremental tensor allocation. No backward graphs are built.', '',
              '| N | Mask | Method | Median ms | Total peak RSS MiB |',
              '| ---: | --- | --- | ---: | ---: |']
    for r in memory:
        lines.append(f"| {r['length']} | {'causal' if r['causal'] else 'noncausal'} | {r['method']} | {r['median_ms']:.3f} | {r['process_peak_rss_bytes']/2**20:.1f} |")
    lines += ['', '![Forward memory and latency](memory.png)', '', '## Tile-size sweep', '',
              'N=512, online kernel; tile sizes 16, 32, 64 and 128. Five warmups, twenty samples.',
              'Larger tiles reduce Python-loop and separate-operation overhead in this implementation.',
              'They can also increase temporary memory. This experiment is not a fused GPU kernel benchmark.', '',
              '| Tile | Mask | Forward ms | Forward + backward ms |',
              '| ---: | --- | ---: | ---: |']
    for tile in (16, 32, 64, 128):
        for causal in (False, True):
            fwd = select(tiles, block_size=tile, causal=causal, phase='forward')[0]
            both = select(tiles, block_size=tile, causal=causal, phase='forward_backward')[0]
            lines.append(f"| {tile} | {'causal' if causal else 'noncausal'} | {fwd['median_ms']:.3f} | {both['median_ms']:.3f} |")
    lines += ['', '![Tile-size comparisons](tiles.png)', '', '## Synthetic training checks', '',
              'One Transformer block with 4x feed-forward width, fixed random features and targets,',
              'MSE and SGD at lr=0.01. Losses below span five warmup plus twenty timed updates.',
              'This measures step cost and numerical behavior, not model quality or convergence.', '',
              '| N | Mask | Method | Initial MSE | Final MSE |',
              '| ---: | --- | --- | ---: | ---: |']
    for r in select(rows, scope='transformer'):
        lines.append(f"| {r['length']} | {'causal' if r['causal'] else 'noncausal'} | {r['method']} | {r['initial_loss']:.6f} | {r['final_loss']:.6f} |")
    lines += ['', '## Reproducibility and limits', '',
              '- Raw samples: `extended_results.json`, `memory_results.json`, `tile_*_results.json`.',
              '- Flat measurements: `results.csv`; charts also exported as SVG.',
              '- Runtime/build details: `hardware.json`; dependencies: `environment.lock.txt`.',
              '- Test output: `test_results.txt`; commands and timing definitions: `README.md`.',
              '- CPU allocations are retained by allocators; process peak memory includes libraries.',
              '- Separate-process startup is excluded from latency but included in peak RSS.',
              '- Twenty samples describe this run, not confidence intervals across independent runs.',
              '- Python tiling adds overhead and retains tile graphs during training.',
              '- No supported GPU is available; no FlashAttention speedup or GPU memory result is claimed.',
              '- These PyTorch results cannot be compared directly with the earlier Ruby FP64 demonstration.', '']
    (ROOT / 'RESULTS.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
