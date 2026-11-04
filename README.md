# CECS530-Project: Self-attention baseline framework

Build a correct reference, measure it, then investigate how attention can use
less memory without changing its mathematical result. The starting experiment
is **forward-only self-attention**, with dropout disabled and no padding. It
benchmarks the attention operation, excluding learned projections and the rest
of a Transformer. Random Q/K/V isolate the kernel; a full self-attention layer
would compute Q=XWq, K=XWk, V=XWv from the same input sequence.

The [measured CPU results](RESULTS.md) cover 104 configurations, including
online attention, backward passes, learned projections, synthetic Transformer
training steps, longer-sequence memory use and a tile-size sweep. All six
correctness tests passed. FlashAttention could not run on this host.

## The mental model

For each query token, ask three questions:

1. **Who is relevant?** Compute query/key dot products, scaled by 1/sqrt(D).
2. **How much weight does each token get?** Apply softmax across the key positions.
3. **What information should be collected?** Average the values using those weights.

In symbols: `O = softmax(Q K^T / sqrt(D) + mask) V`.
The mask is zero for allowed positions and negative infinity for forbidden ones.
A causal mask allows a query at position i to attend only to keys j <= i.
Softmax subtracts the row maximum before exponentiation for numerical stability.

Q/K/V and O use `(B, N, H, D)` here. Internally the dense reference moves heads
ahead of sequence, producing scores of shape `(B, H, N, N)`.
This is the baseline's defining cost: approximately `4 B H N^2 D` FLOPs for
the two dense matrix multiplications, plus softmax and masking. One score
matrix occupies `B H N^2 * bytes_per_element` bytes. Scores, probabilities,
temporaries and autograd state can require more. The dense causal implementation
still computes the full matrix before masking.

A baseline is an **experimental contract**, not merely slow code. Specify shape,
layout, dtype, mask, dropout, device and forward versus training before comparing
methods. Validate values first; use tolerances rather than bitwise equality.

## Files and setup

- `attention.py`: readable PyTorch dense reference and educational online softmax.
- `test_attention.py`: known answers, masking, tile boundaries, gradient agreement
  and a finite-difference gradient check.
- `benchmark.py`: baseline measurement and optional direct FlashAttention comparison.
- `models.py`: learned attention projections and a small Transformer block.
- `benchmark_extended.py`: CPU inference, backward, forward-plus-backward and
  Transformer training-step experiments, with isolated-process peak RSS.
- `run_local.rb`: dependency-free fallback used on the original host.
- `local_results.json`: actual fallback measurements, including raw timing samples.

In an environment with a working Python runtime compatible with your PyTorch
build (the local run uses Python 3.9.6):

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest -v
python benchmark.py --device cpu --dtype float32 --output cpu_results.json
```

Install a PyTorch build suitable for your device. For a supported CUDA machine,
install FlashAttention according to the upstream instructions and run:

```sh
python benchmark.py --device cuda --dtype float16 --flash \
  --lengths 128 256 512 1024 2048 --output gpu_results.json
python -m pip freeze > environment.txt
```

The comparison calls the repository's `flash_attn.flash_attn_func` interface.
The harness fails explicitly if FlashAttention cannot load; it does not silently
replace it with another implementation. Upstream requirements and supported
devices can change; consult the [installation instructions](https://github.com/Dao-AILab/flash-attention#installation-and-features).
Record the upstream commit as well if installing from source.

The benchmark uses identical Q/K/V for both methods, seed 17, both causal modes,
five warmups and twenty recorded samples. Allocation of inputs and the FP32
correctness oracle are outside the timing interval. CUDA events synchronize
each timed sample. TF32 is disabled. Inference mode excludes autograd overhead.
CUDA memory is reported as incremental peak PyTorch allocated memory above the
already-live tensors, including output and temporaries. It excludes allocator
reservation, inputs, oracle and CUDA context, so it is not total process memory.
This original harness does not measure CPU peak memory or backward performance;
use `benchmark_extended.py` for those measurements.

The oracle uses FP32 versions of the same rounded inputs. The harness records
max absolute error and relative L2 error and applies dtype-dependent tolerances.
These tolerances are starting settings for random inputs, not numerical
guarantees for every distribution. Study large scores and adversarial inputs
as an extension. Neither the reference nor the exercises support fully masked
rows, padding, unequal sequence lengths or grouped-query attention yet.

## What FlashAttention changes

The upstream project contains both an explicit
[`SelfAttention` reference](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/modules/mha.py)
and the GPU-backed
[`flash_attn_func` interface](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_interface.py).
Our baseline follows the same score → softmax → value aggregation structure.
It uses an exact negative-infinity causal mask; the upstream module reference
uses a large finite negative mask. The fixed equal-length, dropout-free
experiment keeps the semantic comparison straightforward.

FlashAttention tiles the computation and carries a running row maximum,
normalizer and weighted-value sum. It avoids storing the full attention matrix
in GPU global memory. For one new key tile with scores S and values Vt:

```text
m_new   = max(m, rowmax(S))
alpha   = exp(m - m_new)
P_tile  = exp(S - m_new)
l_new   = alpha * l + rowsum(P_tile)
acc_new = alpha * acc + P_tile @ Vt
O       = acc / l                         # after all tiles
```

The rescaling preserves contributions from earlier tiles when the maximum
changes. This computes the same mathematical attention, subject to floating
point differences. Dense attention still has quadratic arithmetic, while the
full quadratic intermediate need not be stored. See the
[FlashAttention paper](https://arxiv.org/abs/2205.14135).

`online_attention` illustrates this recurrence in Python. It launches separate
tensor operations inside loops; FlashAttention implements hardware-optimized
fused kernels. Our teaching version's **forward inference** avoids full score
storage, but its automatic backward retains tile graphs. It is not a
memory-efficient training implementation. FlashAttention's backward instead
recomputes attention quantities; implementing that is an advanced extension.

The dense and online implementations have the same attention feature scope:
equal-length multihead attention, causal/noncausal masking, and autograd.
The model wrappers can use either implementation with the same learned
projections and Transformer structure. Neither implementation currently adds
padding masks, dropout, grouped-query attention or an API to return attention
weights. Dense attention's full score/probability intermediates are easier to
inspect while debugging; online attention instead has tile-size control.
Its score temporaries are bounded by the tile size, so a tile smaller than N
avoids constructing the full N-by-N score matrix during inference.

## What actually ran on the original host

Host: macOS 12.7.6, x86_64, Intel Iris Graphics 6100. System Python could not
start because its developer tools were missing. Shell network DNS prevented
downloading a replacement runtime. No Python dependencies were installed.

The Ruby fallback runs the same dense attention equations with double-precision
numbers, B=H=1, D=32, sequence lengths 31/64/128/256, two warmups and seven
samples. Run it again with `ruby run_local.rb`.

It checks independent known answers, dense/online agreement for multiple tile
sizes, partial tiles and causal invariance under changes to future keys/values.
It does not validate PyTorch code or gradients. Its timings include interpreted
scalar loops and must not be compared with GPU kernel timings. The recorded
matrix sizes describe numeric payload only, excluding Ruby object overhead.

Measured dense forward medians (milliseconds):

| N | Noncausal | Causal | One FP64 score matrix payload |
| ---: | ---: | ---: | ---: |
| 31 | 7.53 | 6.22 | 7,688 bytes |
| 64 | 34.39 | 28.72 | 32 KiB |
| 128 | 133.09 | 111.93 | 128 KiB |
| 256 | 514.42 | 447.60 | 512 KiB |

The largest dense/online error was 2.78e-16. The Ruby fallback skips causal
masked dot products, whereas the PyTorch dense reference computes and then
masks them; causal timings between these implementations are not comparable.
The approximately fourfold growth from N=64 to 128 to 256 is consistent with
quadratic work in this small experiment. It is not a GPU performance prediction.

That initial run did not execute Python or FlashAttention. After installing
Apple's Command Line Tools, Python became available. The extended CPU experiment
below supersedes the Python availability limitation. FlashAttention still
cannot execute on this machine; no measured FlashAttention speedup is available.

## Extended CPU experiments

The extended harness measures both implementations in FP32 at B=1, H=4, D=64,
N=128/256/512, causal and noncausal, using one CPU thread and 32-token tiles.
Every configuration gets five warmups and twenty timed repetitions in a new
process. The subprocess startup time is excluded from latency.

```sh
source .venv/bin/activate
# On this Intel Mac, use the tested compatible dependency set:
python -m pip install -r requirements-macos-intel.txt
python -m unittest -v
python benchmark_extended.py --output extended_results.json
```

`kernel` accepts precomputed Q/K/V. `layer` includes Q/K/V and output projections.
Both measure inference forward, backward alone, and forward-plus-backward.
Backward-alone preparation builds a fresh forward graph outside the timer;
gradient clearing is also outside the timer for both backward modes.
Do not add inference-forward and backward medians and expect the combined
measurement: inference omits graph construction, and each phase is independent.

`transformer/train_step` measures one pre-norm Transformer block with two
residual connections, LayerNorm, attention and a GELU feed-forward network with
4x hidden width. A timed step includes zeroing parameter gradients, forward,
MSE against a fixed random target, backward, and an SGD parameter update at
learning rate 0.01. Warmups also update parameters. This is synthetic training
on fixed feature tensors, with no tokenizer, embedding, vocabulary projection or
data loader. Its loss trajectory does not measure text quality.

All saved rows contain latency samples, finite-output/gradient status and peak
resident memory. `process_peak_rss_bytes` is the isolated process's lifetime peak
RSS, including Python, PyTorch, inputs, weights, gradients, warmups and final
checks. It is not incremental tensor allocation, and small memory differences
are not evidence of an algorithmic memory advantage. CPU allocator retention
and library initialization influence it. Automatic differentiation retains
online attention's per-tile graphs, so online training can use more memory.

Larger forward-only memory cases and tile-size experiments use the same harness:

```sh
python benchmark_extended.py --lengths 2048 4096 --scopes kernel --phases forward \
  --warmup 1 --repeats 5 --output memory_results.json
for tile in 16 64 128; do
  python benchmark_extended.py --lengths 512 --scopes kernel \
    --phases forward forward_backward --methods online --block-size "$tile" \
    --output "tile_${tile}_results.json"
done
```

The primary run already measures tile size 32. See `test_results.txt`,
`hardware.json`, the raw JSON files and the generated results summary for
validation and environment details. Keep the Ruby results separate because
they use a different language, precision and head configuration.

Run `python summarize_results.py` after the experiments to generate `RESULTS.md`,
`results.csv`, and PNG/SVG charts. Alternatively, `sh run_cpu_suite.sh` runs the
tests, primary benchmark, memory cases, tile sweep and summary in sequence.
Plotting requires Matplotlib, included in `requirements-macos-intel.txt`; on
other platforms install a compatible Matplotlib version alongside PyTorch.

## Suggested project milestones

1. **Establish correctness.** Draw a three-token attention matrix, derive the
   uniform-score result and causal prefix averages, and run the tests. Explain
   why Q/K dot products need scaling and why softmax operates over keys.
2. **Establish the baseline.** Fix B/H/D, sweep N, save raw samples and plot median
   latency. Explain why doubling N predicts four times the matrix storage.
   Keep the device and precision fixed when interpreting timing trends.
3. **Implement an optimization.** Start with `online_attention`; vary tile size
   and compare outputs and gradients. Describe its working memory. Explain why
   mathematical equivalence alone does not guarantee a faster implementation.
4. **Compare with FlashAttention.** Run `--flash` on one supported GPU. For each
   matching configuration compute `dense median_ms / flash median_ms`. Plot
   latency, measured incremental memory and numerical error against N. Repeat
   runs to assess variation; retain failures and out-of-memory cases in the report.
5. **Choose one extension.** Add a padding mask with explicit fully masked-row
   behavior, train a tiny Transformer, implement a fused kernel, or add a separate
   forward-plus-backward benchmark. Do not mix training and inference results.

Submit the hypothesis, equations, implementation changes, passing test output,
environment and dependency versions, raw results, plots and a discussion of
limitations. Separate analytical predictions from measurements. A defensible
result that an attempted optimization is slower is a valid project outcome.

Use this as starter code subject to your course's rules, and credit the upstream
paper and implementations when using their ideas or code.
