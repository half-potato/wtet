# Building Python Bindings

Quick guide to build and test the Python bindings.

## Prerequisites

```bash
# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install Python 3.8+ and pip
# (varies by platform)

# Install maturin
pip install maturin
```

## Build Commands

### Development Build (Editable Install)

For development, build and install in editable mode:

```bash
# Set compatibility flag for Python 3.14+
export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1

# Build and install (editable)
maturin develop --release --features python
```

This creates an editable install in your current Python environment. Changes to Python wrapper code will be reflected immediately, but Rust changes require rebuilding.

### Production Build (Wheel)

To create a distributable wheel:

```bash
export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
maturin build --release --features python
```

The wheel will be created in `target/wheels/`. Install it with:

```bash
pip install target/wheels/gdel3d-*.whl
```

### Quick Test

After building:

```bash
# Test import
python -c "import gdel3d; print(gdel3d.__version__)"

# Run basic example
python python/examples/basic_example.py

# Run tests
pip install pytest
pytest python/tests/ -v
```

## Build Options

### Debug Build (Faster Compile, Slower Runtime)

```bash
maturin develop --features python
```

### Release Build (Optimized, Recommended)

```bash
maturin develop --release --features python
```

### Specific Python Version

```bash
# Use specific Python interpreter
maturin develop --release --features python -i python3.11
```

## Environment Variables

- `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` - Allow Python versions newer than officially supported
- `RUST_LOG=debug` - Enable debug logging during runtime

## Troubleshooting

### "Failed to find GPU adapter"

Ensure your system has:
- A compatible GPU (supports Vulkan, Metal, or DirectX 12)
- Up-to-date GPU drivers

### Import Error

If `import gdel3d` fails:
```bash
# Rebuild with verbose output
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop --release --features python -v
```

### Python Version Issues

If you get PyO3 version errors:
```bash
# Check Python version
python --version

# Set forward compatibility
export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
```

## Performance Tips

1. **Always use release builds** for benchmarking (`--release` flag)
2. **Use float32 dtype** for input points (not float64)
3. **Pre-allocate arrays** when possible
4. **GPU initialization** happens on first call - subsequent calls are faster

## Next Steps

- See [PYTHON_BINDINGS.md](./PYTHON_BINDINGS.md) for full API documentation
- Run `python python/examples/benchmark.py` for performance testing
- Check `python/tests/` for usage examples
