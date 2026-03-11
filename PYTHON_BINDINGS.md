# Python Bindings for gDel3D

This document describes how to build, install, and use the Python bindings for the GPU-accelerated 3D Delaunay triangulation library.

## Requirements

- Python 3.8 or higher
- NumPy 1.20 or higher
- Rust toolchain (for building from source)
- GPU with wgpu support (Vulkan, Metal, or DX12)

## Installation

### From Source (Development)

1. Install maturin:
   ```bash
   pip install maturin
   ```

2. Build and install in development mode:
   ```bash
   maturin develop --release --features python
   ```

3. Test the installation:
   ```bash
   python -c "import gdel3d; print(gdel3d.__version__)"
   ```

### Build Wheel for Distribution

```bash
maturin build --release --features python
pip install target/wheels/gdel3d-*.whl
```

## Quick Start

```python
import numpy as np
import gdel3d

# Generate random 3D points
points = np.random.rand(1000, 3).astype(np.float32)

# Compute Delaunay triangulation
result = gdel3d.delaunay(points)

print(f"Generated {result.num_tets} tetrahedra")
print(f"Failed to insert {result.num_failed} vertices")

# Access results
tets = result.tets              # (M, 4) array of vertex indices
adjacency = result.adjacency    # (M, 4) array of packed adjacency
failed = result.failed_verts    # (K,) array of failed vertex indices
```

## Configuration

Customize the triangulation algorithm:

```python
config = gdel3d.Config(
    insertion_rule="circumcenter",  # or "centroid"
    enable_flipping=True,           # Enable Delaunay flipping
    enable_sorting=False,           # Enable Morton sorting
    enable_hilbert_sorting=False,   # Enable Hilbert sorting
    enable_splaying=True,           # Enable star splaying
    max_insert_iterations=100,      # Max insertion iterations
    max_flip_iterations=10,         # Max flipping iterations per round
)

result = gdel3d.delaunay(points, config)
```

## API Reference

### Functions

#### `delaunay(points, config=None)`

Compute 3D Delaunay triangulation of points.

**Parameters:**
- `points`: NumPy array of shape `(N, 3)` and dtype `float32`
- `config`: Optional `Config` object (defaults to `Config()`)

**Returns:** `DelaunayResult` object

**Raises:**
- `ValueError`: If points have wrong shape or fewer than 4 points
- `RuntimeError`: If GPU initialization fails

---

#### `decode_adjacency(packed)`

Decode packed adjacency value into `(tet_idx, face_idx)`.

**Parameters:**
- `packed`: `uint32` packed adjacency value

**Returns:** Tuple `(tet_idx, face_idx)`

---

#### `encode_adjacency(tet_idx, face_idx)`

Encode `(tet_idx, face_idx)` into packed adjacency value.

**Parameters:**
- `tet_idx`: Tetrahedron index
- `face_idx`: Face index (0-3)

**Returns:** `uint32` packed adjacency value

---

#### `initialize_gpu()`

Explicitly initialize GPU device. Auto-called on first `delaunay()` call.

**Returns:** String describing GPU device

---

#### `gpu_info()`

Get GPU device information.

**Returns:** String with GPU info, or message if not initialized

---

### Classes

#### `Config`

Configuration for Delaunay triangulation algorithm.

**Constructor:**
```python
Config(
    insertion_rule="circumcenter",
    enable_flipping=True,
    enable_sorting=False,
    enable_hilbert_sorting=False,
    enable_splaying=True,
    max_insert_iterations=100,
    max_flip_iterations=10,
)
```

---

#### `DelaunayResult`

Result of Delaunay triangulation.

**Attributes:**
- `tets`: NumPy array of shape `(M, 4)`, dtype `uint32` - vertex indices for each tetrahedron
- `adjacency`: NumPy array of shape `(M, 4)`, dtype `uint32` - packed adjacency (tet_idx, face_idx)
- `failed_verts`: NumPy array of shape `(K,)`, dtype `uint32` - indices of vertices that failed to insert

**Properties:**
- `num_tets`: Number of tetrahedra (same as `len(tets)`)
- `num_failed`: Number of failed vertices (same as `len(failed_verts)`)

---

### Constants

- `INVALID`: Value `2^32 - 1` representing no adjacency (boundary face)
- `__version__`: Package version string

---

## Adjacency Format

Adjacency information is stored in a packed format using 5-bit encoding:

```
packed_value = (tet_idx << 5) | face_idx
```

- **Bits 0-1**: Face index (0-3)
- **Bits 5-31**: Tetrahedron index

Use `decode_adjacency()` and `encode_adjacency()` helper functions.

**Example:**
```python
# Get adjacency for face 2 of tetrahedron 0
adj = result.adjacency[0, 2]

if adj != gdel3d.INVALID:
    neighbor_tet, neighbor_face = gdel3d.decode_adjacency(adj)
    print(f"Face 2 connects to tet {neighbor_tet}, face {neighbor_face}")
else:
    print("Face 2 is on the boundary")
```

---

## Examples

See `python/examples/` directory:

- `basic_example.py` - Basic usage and adjacency decoding
- `benchmark.py` - Performance benchmarking

Run examples:
```bash
python python/examples/basic_example.py
python python/examples/benchmark.py
```

---

## Testing

Run Python tests:
```bash
# Install pytest
pip install pytest

# Run all tests
pytest python/tests/ -v

# Run specific test
pytest python/tests/test_basic.py::test_cube -v
```

---

## Performance

Typical performance on modern GPU (tested on desktop GPU):

| Points   | Time (ms) | Throughput (points/sec) |
|----------|-----------|-------------------------|
| 1K       | ~5        | ~200K                   |
| 10K      | ~20       | ~500K                   |
| 100K     | ~100      | ~1M                     |
| 1M       | ~200      | ~5M                     |

*Results vary based on GPU, point distribution, and configuration.*

---

## Troubleshooting

### GPU not found

If you get "Failed to find GPU adapter":
- Ensure your system has a compatible GPU (Vulkan, Metal, or DX12)
- Update GPU drivers
- Check `wgpu` backend support for your platform

### Import error

If `import gdel3d` fails:
- Ensure you built with `--features python`
- Check maturin installation: `pip install maturin`
- Try rebuilding: `maturin develop --release --features python`

### Wrong NumPy dtype

The library requires `float32` dtype:
```python
# Wrong
points = np.random.rand(100, 3)  # dtype=float64

# Correct
points = np.random.rand(100, 3).astype(np.float32)
```

---

## License

Same as parent project (MIT OR Apache-2.0).
