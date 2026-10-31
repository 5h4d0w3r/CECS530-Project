# Dependency-free fallback for this host. Ruby Float is double precision.
# Mirrors the attention equations; does NOT execute the Python/PyTorch files.
require 'json'

def dot(a, b)
  a.each_index.inject(0.0) { |sum, i| sum + a[i] * b[i] }
end

def dense(q, k, v, causal)
  n, d = q.length, q[0].length
  scores = q.each_with_index.map do |row, i|
    k.each_with_index.map { |key, j| causal && j > i ? -Float::INFINITY : dot(row, key) / Math.sqrt(d) }
  end
  probs = scores.map do |row|
    m = row.max
    exps = row.map { |s| Math.exp(s - m) }
    z = exps.inject(:+)
    exps.map { |e| e / z }
  end
  probs.map { |row| Array.new(d) { |c| n.times.inject(0.0) { |s, j| s + row[j] * v[j][c] } } }
end

def online(q, k, v, causal, block = 32)
  n, d = q.length, q[0].length
  q.each_with_index.map do |row, i|
    m, z, acc = -Float::INFINITY, 0.0, Array.new(d, 0.0)
    limit = causal ? i + 1 : n
    (0...limit).step(block) do |start|
      indices = (start...[start + block, limit].min).to_a
      scores = indices.map { |j| dot(row, k[j]) / Math.sqrt(d) }
      next_m = [m, scores.max].max
      alpha = Math.exp(m - next_m)
      weights = scores.map { |s| Math.exp(s - next_m) }
      acc.map! { |a| a * alpha }
      indices.each_with_index { |j, t| d.times { |c| acc[c] += weights[t] * v[j][c] } }
      z = alpha * z + weights.inject(:+)
      m = next_m
    end
    acc.map { |a| a / z }
  end
end

def error(a, b)
  a.flatten.zip(b.flatten).map { |x, y| (x - y).abs }.max
end

def assert_close(a, b)
  raise "incorrect result: #{error(a, b)}" unless error(a, b) < 1e-10
end

def measure
  2.times { yield }
  samples = 7.times.map do
    GC.start
    start = Process.clock_gettime(Process::CLOCK_MONOTONIC)
    yield
    (Process.clock_gettime(Process::CLOCK_MONOTONIC) - start) * 1000
  end
  {median_ms: samples.sort[3], min_ms: samples.min, max_ms: samples.max, samples_ms: samples}
end

# Independent known answer and causal prefix averages.
q = Array.new(3) { [0.0, 0.0] }
k = Array.new(3) { [1.0, 1.0] }
v = [[1.0, 2.0], [3.0, 4.0], [8.0, 9.0]]
[:dense, :online].each do |method|
  assert_close(send(method, q, k, v, false), Array.new(3) { [4.0, 5.0] })
  assert_close(send(method, q, k, v, true), [[1.0, 2.0], [2.0, 3.0], [4.0, 5.0]])
end
rng = Random.new(17)
rows = []
[31, 64, 128, 256].each do |n|
  d = 32
  q, k, v = Array.new(3) { Array.new(n) { Array.new(d) { rng.rand * 2 - 1 } } }
  [false, true].each do |causal|
    reference = dense(q, k, v, causal)
    e = error(reference, online(q, k, v, causal))
    raise 'online mismatch' unless e < 1e-10
    # Uneven blocks, single-element blocks, and a single whole-sequence block.
    [1, 7, n + 1].each { |tile| assert_close(reference, online(q, k, v, causal, tile)) }
    k2, v2 = k.map(&:dup), v.map(&:dup)
    cut = n / 2
    (cut...n).each { |j| k2[j].map! { |x| x + 100 }; v2[j].map! { |x| x - 100 } }
    [:dense, :online].each do |method|
      before = send(method, q, k, v, true)
      after = send(method, q, k2, v2, true)
      assert_close(before.take(cut), after.take(cut))
    end
    row = {batch: 1, heads: 1, length: n, dim: d, causal: causal,
           max_abs_error_online_vs_dense: e, dense: measure { dense(q, k, v, causal) },
           one_score_matrix_numeric_payload_bytes: n * n * 8}
    rows << row
    puts JSON.generate(row)
  end
end
report = {runtime: RUBY_DESCRIPTION, platform: RUBY_PLATFORM, seed: 17,
          dtype: 'Ruby Float (FP64)', warmup: 2, repeats: 7,
          scope: 'forward CPU interpreter demonstration, not GPU performance',
          checks: 'known answers, dense/online agreement across tiles, causal future invariance passed',
          memory_note: 'numeric payload only; Ruby object overhead and process peak not measured',
          flash_attention_executed: false, python_executed: false, rows: rows}
File.write(File.join(__dir__, 'local_results.json'), JSON.pretty_generate(report) + "\n")
