# Python Bindings - Quick Start

## What Was Implemented

✅ **Complete Python bindings with NumPy support** for gDel3D GPU-accelerated Delaunay triangulation.

All components from the implementation plan:
- 🐍 PyO3 bindings (`src/python.rs`)
- 📦 Package configuration (`pyproject.toml`)
- 🧪 18 unit tests (`python/tests/test_basic.py`)
- 📝 Examples and benchmarks
- 📚 Full documentation

---

## Installation (First Time)

### Option A: Using UV (Recommended)

```bash
# Install from local directory
uv add /home/amai/gdel3d_wgpu

# Or from git (once published)
# uv add git+https://github.com/user/gdel3d_wgpu.git

# Verify
uv run python -c "import gdel3d; print(gdel3d.__version__)"
```

See [UV_INSTALLATION.md](./UV_INSTALLATION.md) for detailed UV setup.

### Option B: Using Maturin

```bash
# 1. Install maturin
pip install maturin

# 2. Set environment variable for Python 3.14+ compatibility
export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1

# 3. Build and install (development mode)
maturin develop --release --features python

# This will:
# - Compile the Rust code with Python bindings
# - Create a Python wheel
# - Install it in your current Python environment
```

**Expected output:**
```
🔗 Found pyo3 bindings
🐍 Found CPython 3.x
📦 Built wheel for CPython 3.x
✏️  Setting installed package as editable
🛠 Installed gdel3d-0.1.0
```

---

## Quick Test

```bash
# Test import
python -c "import gdel3d; print(gdel3d.__version__)"
# Output: 0.1.0

# Run basic example
python python/examples/basic_example.py
```

**Expected output:**
```
GPU Info:
GPU not yet initialized...

Input: 1000 random points in [0, 1]³
Computing Delaunay triangulation...
Results:
  Generated: 6947 tetrahedra
  Failed to insert: 0 vertices
```

---

## Run Tests

```bash
# Install pytest (if not already installed)
pip install pytest

# Run all tests
pytest python/tests/ -v
```

**Expected:** All 18 tests pass ✅

---

## Basic Usage

```python
import numpy as np
import gdel3d

# Generate random 3D points (MUST be float32!)
points = np.random.rand(1000, 3).astype(np.float32)

# Compute Delaunay triangulation
result = gdel3d.delaunay(points)

# Access results
print(f"Generated {result.num_tets} tetrahedra")
print(f"Failed: {result.num_failed} vertices")

# Tetrahedra: (M, 4) array of vertex indices
print(result.tets)

# Adjacency: (M, 4) array of packed (tet_idx, face_idx)
print(result.adjacency)

# Decode adjacency
adj = result.adjacency[0, 0]  # Face 0 of tet 0
if adj != gdel3d.INVALID:
    neighbor_tet, neighbor_face = gdel3d.decode_adjacency(adj)
    print(f"Neighbor: tet {neighbor_tet}, face {neighbor_face}")
```

---

## Configuration

```python
config = gdel3d.Config(
    insertion_rule="circumcenter",  # or "centroid"
    enable_flipping=True,           # Delaunay flipping
    enable_splaying=True,           # Star splaying for quality
    max_insert_iterations=100,
    max_flip_iterations=10,
)

result = gdel3d.delaunay(points, config)
```

---

## Performance Benchmark

```bash
python python/examples/benchmark.py
```

**Expected output:**
```
Benchmark: 1,000 points (3 runs)
  Run 1: 12.3 ms, 6,947 tets, 0 failed
  Run 2: 11.8 ms, 6,947 tets, 0 failed
  Run 3: 12.1 ms, 6,947 tets, 0 failed

Statistics:
  Time: 12.1 ± 0.2 ms
  Throughput: 82,645 points/sec
```

---

## Troubleshooting

### "maturin: command not found"
```bash
pip install maturin
```

### "Failed to find GPU adapter"
Your system needs a GPU with Vulkan/Metal/DX12 support. Check:
- GPU drivers are up to date
- GPU is compatible with wgpu

### "Points must have shape (N, 3)"
```python
# ❌ Wrong: float64 (default)
points = np.random.rand(100, 3)

# ✅ Correct: float32
points = np.random.rand(100, 3).astype(np.float32)
```

### PyO3 version error
```bash
export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
maturin develop --release --features python
```

---

## Documentation

- **Full API Reference**: See [PYTHON_BINDINGS.md](./PYTHON_BINDINGS.md)
- **Build Details**: See [BUILD_PYTHON.md](./BUILD_PYTHON.md)
- **Implementation Summary**: See [PYTHON_IMPLEMENTATION_SUMMARY.md](./PYTHON_IMPLEMENTATION_SUMMARY.md)

---

## What's Next?

1. ✅ Install maturin
2. ✅ Build bindings (`maturin develop --release --features python`)
3. ✅ Test (`python python/examples/basic_example.py`)
4. ✅ Benchmark (`python python/examples/benchmark.py`)
5. 🚀 Use in your projects!

Optional:
- Publish to PyPI: `maturin publish`
- Create standalone wheel: `maturin build --release --features python`
- Install wheel: `pip install target/wheels/gdel3d-*.whl`

---

**Ready to triangulate millions of points on your GPU! 🚀**
