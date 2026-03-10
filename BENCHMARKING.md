# Benchmarking Guide

## Quick Start

### Run Full Benchmark Suite
```bash
./benchmark.sh
```

### Run Individual Tests
```bash
# Small (20k points, ~0.3s)
cargo test --release --lib test_delaunay_uniform_20k -- --nocapture

# Medium (200k points, ~1s)
cargo test --release --lib test_delaunay_uniform_200k -- --nocapture

# Large (2M points, ~13s)
cargo test --release --lib test_delaunay_uniform_2M -- --nocapture
```

---

## What to Look For

### 1. Total Runtime
```
[TIMING] Phase 1 total: 1.87 seconds
```

### 2. CPU Profiler Breakdown
```
9_compact         | calls:    38 | total:   988.59 ms
5_flip_fast       | calls:    38 | total:   136.27 ms
1_vote_phase      | calls:    38 | total:   102.38 ms
```

### 3. Adaptive Mode Switching
```
[FLIP] Iteration 1: mode=MarkCompact, input=880, output=0
[FLIP] Iteration 2: mode=CollectCompact, input=216, output=0
```

**Expected behavior:**
- Large queues (≥256): Use `MarkCompact`
- Small queues (<256): Use `CollectCompact`
- Tail iterations should all be `CollectCompact`

### 4. Insertion Success Rate
```
[INFO] Phase 1 complete after 38 iterations, all points inserted
```

Should see "all points inserted" for uniform distributions.

---

## Performance Expectations

### Dataset Sizes

| Points | Expected Time | Expected Tets | Iterations |
|--------|---------------|---------------|------------|
| 20k    | ~0.3s        | ~130k         | ~20        |
| 200k   | ~1.0s        | ~1.3M         | ~30        |
| 2M     | ~1.8s        | ~13M          | ~38        |

**Note:** Times are for NVIDIA RTX 3090. Your results may vary based on GPU.

### Phase Breakdown (2M points)

| Phase            | Time   | Percentage |
|------------------|--------|------------|
| Compaction       | 988ms  | 53%        |
| Flipping         | 136ms  | 7%         |
| Vote/Locate      | 102ms  | 5%         |
| Split            | 128ms  | 7%         |
| Other            | 500ms  | 28%        |

---

## Optimization Verification

### Adaptive Flip Compaction

To verify the optimization is working:

```bash
cargo test --release --lib test_delaunay_uniform_200k -- --nocapture 2>&1 | grep "mode="
```

**Expected output:**
```
[FLIP] Iteration 1: mode=MarkCompact, input=141172, output=0
...
[FLIP] Iteration 25: mode=MarkCompact, input=460, output=0
[FLIP] Iteration 26: mode=CollectCompact, input=216, output=0  ← Transition
[FLIP] Iteration 27: mode=CollectCompact, input=136, output=0
```

**Success criteria:**
- Mode switches from `MarkCompact` → `CollectCompact` when queue < 256
- Last 5-10 iterations should be `CollectCompact`
- GPU passes saved = (CollectCompact iterations) × 2

---

## Profiling GPU Performance

### Enable GPU Timestamps (if supported)

GPU profiling is automatically enabled if your device supports `TIMESTAMP_QUERY`.

Look for:
```
[PROFILER] GPU profiling enabled
```

If you see:
```
[PROFILER] TIMESTAMP_QUERY not supported, GPU profiling disabled
```

Then your device doesn't support GPU timestamps (common on some integrated GPUs).

### CPU vs GPU Time

- **CPU time**: Includes submit/poll/readback overhead
- **GPU time**: Pure shader execution time (if timestamps supported)

CPU time is usually 2-3× higher due to synchronization overhead.

---

## Comparing Against CUDA

If you have the original CUDA implementation compiled:

```bash
# Run CUDA version
cd gDel3D/GDelFlipping
./build/gdel3d --points 200000

# Run WGPU version
cd /home/amai/gdel3d_wgpu
cargo test --release --lib test_delaunay_uniform_200k -- --nocapture
```

**Expected results:**
- WGPU should be within 2-3× of CUDA performance
- Iteration counts should be identical
- Output tet counts should match exactly

---

## Memory Usage

### Check Buffer Sizes

```bash
cargo test --release --lib test_delaunay_uniform_2M -- --nocapture 2>&1 | grep "GPU STATE"
```

Look for:
```
[GPU STATE] Input: 2000000 points
[GPU STATE] Creating buffers...
```

### Maximum Supported Dataset

- **2M points**: ✅ Fully supported (at WGPU 256 MB buffer limit)
- **20M points**: ❌ Exceeds WGPU limits (requires chunking)

---

## Troubleshooting

### Test Fails with "Binding size overflow"

If you see:
```
Buffer binding with size X would overflow buffer size Y
```

This means a recent code change broke the buffer capping logic. Check:
1. `dispatch_update_opp()` - Should cap flip array bindings
2. `dispatch_flip()` - Should cap tet buffer bindings
3. Any new dispatch methods - Must cap to `max_tets` or `max_flips`

### Performance Regression

If performance suddenly drops:

1. **Check optimization level:**
   ```bash
   # Should always use --release
   cargo test --release --lib test_delaunay_uniform_200k
   ```

2. **Check adaptive mode is working:**
   ```bash
   cargo test --release --lib test_delaunay_uniform_200k -- --nocapture 2>&1 | grep "mode=" | tail -10
   ```

   Should see `CollectCompact` in tail iterations.

3. **Check iteration count:**
   - If iteration count increased significantly → insertion convergence issue
   - If iteration count normal but slower → GPU kernel issue

### Out of Memory

If you see VRAM-related errors:

1. Check dataset size doesn't exceed 2M points
2. Check `max_tets` calculation in `mod.rs`
3. Reduce dataset size or enable chunking

---

## Custom Benchmarks

### Test Your Own Point Cloud

```rust
// In tests.rs
#[test]
fn test_custom_points() {
    let points: Vec<[f32; 3]> = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        // ... your points
    ];

    let config = GDelConfig::default();
    let result = run_delaunay(&points, &config);

    println!("Tets: {}", result.tets.len());
}
```

### Measure Specific Phases

Add timing around specific operations:

```rust
let start = std::time::Instant::now();
// ... operation ...
eprintln!("[TIMING] Operation took: {:.2}ms", start.elapsed().as_secs_f64() * 1000.0);
```

---

## CI/CD Integration

### GitHub Actions

```yaml
- name: Run benchmarks
  run: ./benchmark.sh > benchmark_results.txt

- name: Upload results
  uses: actions/upload-artifact@v3
  with:
    name: benchmark-results
    path: benchmark_results.txt
```

### Performance Regression Detection

Compare against baseline:

```bash
# Save baseline
cargo test --release --lib test_delaunay_uniform_200k -- --nocapture 2>&1 |
  grep "Phase 1 total" > baseline.txt

# Compare after changes
cargo test --release --lib test_delaunay_uniform_200k -- --nocapture 2>&1 |
  grep "Phase 1 total" > current.txt

diff baseline.txt current.txt
```

---

## Further Reading

- See `ADAPTIVE_MODE_IMPLEMENTATION.md` for flip compaction optimization details
- See `BUFFER_LIMIT_FIXES.md` for buffer size limit handling
- See CUDA reference: `gDel3D/GDelFlipping/src/gDel3D/GpuDelaunay.cu`
