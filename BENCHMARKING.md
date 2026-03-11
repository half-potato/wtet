# Benchmarking gDel3D

This document describes the various benchmarking tools available for testing performance.

## Quick Summary

We have **3 benchmarking options**:

1. **Rust Benchmark Binary** (`examples/benchmark.rs`) - ⭐ **Recommended**
2. **Rust Benchmark Test** (`test benchmark_200k_multi_seed`)
3. **Python Benchmark** (`python/examples/benchmark.py`)

---

## 1. Rust Benchmark Binary (Recommended)

A standalone benchmark program with multiple test modes.

### Usage

```bash
# Default: 100, 1k, 10k, 100k, 200k points
cargo run --release --example benchmark

# Quick test: 100, 1k, 10k points
cargo run --release --example benchmark -- quick

# Full test: up to 1M points
cargo run --release --example benchmark -- full

# Extreme test: up to 2M points
cargo run --release --example benchmark -- extreme

# Custom size: 500k points
cargo run --release --example benchmark -- 500000
```

### Example Output

```
╔═══════════════════════════════════════════════════════════════════╗
║         gDel3D GPU Delaunay Triangulation Benchmark              ║
╚═══════════════════════════════════════════════════════════════════╝

Initializing GPU...
GPU: NVIDIA GeForce RTX 3080 (Vulkan)
Device created successfully

Configuration:
  Insertion rule:   Circumcenter
  Flipping:         true
  Splaying:         true
  Max iterations:   100

======================================================================
Benchmark:        100 points (5 runs)
======================================================================
  Run  1/5:     3.2 ms |       84 tets |    0 failed
  Run  2/5:     3.1 ms |       84 tets |    0 failed
  Run  3/5:     3.0 ms |       84 tets |    0 failed
  Run  4/5:     3.1 ms |       84 tets |    0 failed
  Run  5/5:     3.2 ms |       84 tets |    0 failed

Statistics:
  Time:         3.1 ± 0.1 ms
  Range:        3.0 - 3.2 ms
  Throughput:  32,258 points/sec
  Avg tets:        84
  Total fail:       0

======================================================================
Benchmark:         1k points (5 runs)
======================================================================
  [...]

======================================================================
Summary
======================================================================
Points          |     Avg Time |      Throughput
----------------+-------------+----------------
100             |        3.1 ms |      32,258/sec
1k              |       10.5 ms |      95,238/sec
10k             |       45.2 ms |     221,239/sec
100k            |      198.7 ms |     503,279/sec
200k            |      401.3 ms |     498,380/sec

======================================================================
Benchmark complete!
======================================================================
```

### Features

- ✅ Multiple dataset sizes
- ✅ Multiple runs per size (3-5 runs)
- ✅ Statistical analysis (mean, std dev, min/max)
- ✅ Throughput calculation
- ✅ Summary table
- ✅ Different seeds per run for robustness
- ✅ Clean formatted output

---

## 2. Rust Benchmark Test

A test function that runs 200k points with 10 different seeds.

### Usage

```bash
# Run the benchmark test (takes ~30 seconds)
cargo test --release benchmark_200k_multi_seed -- --nocapture --ignored
```

### Features

- ✅ Fixed seeds for reproducibility
- ✅ 10 iterations with different point distributions
- ✅ Full validation after each run
- ✅ Statistical summary
- ✅ Detailed timing breakdown

### When to Use

- When you want reproducible results with fixed seeds
- When testing specifically 200k points
- When you need validation after each run
- When investigating performance variance

---

## 3. Python Benchmark

Python benchmark using the Python bindings.

### Usage

```bash
# Install Python bindings first
maturin develop --release --features python

# Run Python benchmark
python python/examples/benchmark.py
```

### Features

- ✅ Multiple dataset sizes (100 → 100k by default)
- ✅ NumPy array benchmarking
- ✅ 3 runs per size
- ✅ Statistical summary
- ✅ Python overhead measurement

### When to Use

- When testing the Python bindings performance
- When comparing Python vs Rust overhead
- When integrating with Python workflows
- When you want to extend with custom analysis

---

## Performance Comparison

Typical results on a modern GPU:

| Points  | Time (ms) | Throughput (points/sec) | Tetrahedra |
|---------|-----------|-------------------------|------------|
| 100     | 3-5       | ~30k                    | ~80        |
| 1,000   | 8-12      | ~90k                    | ~900       |
| 10,000  | 35-50     | ~220k                   | ~9k        |
| 100,000 | 150-250   | ~500k                   | ~95k       |
| 200,000 | 300-500   | ~500k                   | ~190k      |
| 1M      | 1500-2500 | ~500k                   | ~950k      |
| 2M      | 3000-5000 | ~500k                   | ~1.9M      |

*Results vary based on GPU, point distribution, and configuration*

---

## Which Benchmark Should I Use?

### Use **Rust Benchmark Binary** if:
- ✅ You want comprehensive testing across multiple sizes
- ✅ You want clean, formatted output
- ✅ You want to quickly test different configurations
- ✅ **This is the recommended option for most use cases**

### Use **Rust Benchmark Test** if:
- You need reproducible results with fixed seeds
- You're investigating a specific performance issue at 200k points
- You want to run benchmarks as part of test suite

### Use **Python Benchmark** if:
- You're testing the Python bindings
- You need to compare Python vs Rust performance
- You want to integrate benchmarking into Python workflows

---

## Interpreting Results

### Good Performance Indicators

- ✅ **Consistent timings** across runs (low std dev)
- ✅ **Zero failed vertices** (all points inserted successfully)
- ✅ **Linear scaling** with point count (roughly)
- ✅ **500k+ points/sec** for large datasets (100k+)

### Red Flags

- ⚠️ **High variance** between runs (>10% std dev)
- ⚠️ **Many failed vertices** (>1% of input)
- ⚠️ **Sudden performance drop** at certain sizes
- ⚠️ **<100k points/sec** for any dataset size

### Optimization Tips

If performance is lower than expected:

1. **Use release mode**: Always benchmark with `--release`
2. **Check GPU load**: Ensure GPU isn't busy with other tasks
3. **Update drivers**: GPU driver version can significantly impact performance
4. **Check configuration**: Disable splaying for speed, enable for quality
5. **Verify data**: Random uniform points are fastest, clustered points slower

---

## Configuration Options

All benchmarks use these default settings:

```rust
GDelConfig {
    insertion_rule: Circumcenter,  // Best quality
    enable_flipping: true,          // Essential for Delaunay
    enable_sorting: false,          // Adds overhead, minimal benefit
    enable_hilbert_sorting: false,  // Experimental
    enable_splaying: true,          // Guarantees Delaunay (adds ~20% overhead)
    max_insert_iterations: 100,     // Usually converges in <20
    max_flip_iterations: 10,        // Usually converges in <5
}
```

To test faster settings (at cost of quality):

```rust
GDelConfig {
    enable_splaying: false,  // Skip CPU splaying (faster, less guaranteed)
    insertion_rule: Centroid, // Slightly faster than Circumcenter
    // ... other settings
}
```

---

## Adding Custom Benchmarks

### Rust Example

```rust
// In examples/benchmark.rs, add a new mode:
let sizes = match benchmark_mode {
    "my_custom" => vec![50_000, 150_000, 250_000],
    // ... existing modes
};
```

Run with:
```bash
cargo run --release --example benchmark -- my_custom
```

### Python Example

```python
# In python/examples/benchmark.py, modify sizes list:
sizes = [5_000, 50_000, 500_000]  # Your custom sizes
```

---

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Benchmark

on: [push]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
      - name: Run benchmark
        run: |
          cargo run --release --example benchmark -- quick > bench_results.txt
          cat bench_results.txt
      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: bench_results.txt
```

---

## Next Steps

- Run `cargo run --release --example benchmark` to see your GPU's performance
- Compare results against expected performance above
- Try different modes (quick, default, full, extreme)
- Modify configuration to test quality vs speed tradeoffs

**Happy benchmarking! 🚀**
