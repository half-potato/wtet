# Installing with UV

This guide shows how to install `gdel3d` using the modern `uv` package manager.

## Prerequisites

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Ensure Rust is installed**:
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

## Installation Methods

### Option 1: From Local Directory

If you have this repository cloned locally:

```bash
# Navigate to your project directory
cd /path/to/your/project

# Add gdel3d from local path
uv add /home/amai/gdel3d_wgpu

# Or from relative path
uv add ../gdel3d_wgpu
```

### Option 2: From Git Repository

If the repository is on GitHub/GitLab:

```bash
# Add from git URL
uv add git+https://github.com/username/gdel3d_wgpu.git

# Or with specific branch/tag
uv add git+https://github.com/username/gdel3d_wgpu.git@main
```

### Option 3: Direct Path in pyproject.toml

Add this to your project's `pyproject.toml`:

```toml
[project]
dependencies = [
    "gdel3d @ file:///home/amai/gdel3d_wgpu",
]
```

Or for git:

```toml
[project]
dependencies = [
    "gdel3d @ git+https://github.com/username/gdel3d_wgpu.git",
]
```

Then run:
```bash
uv sync
```

## Environment Variables

The build process requires setting an environment variable for Python 3.14+ compatibility. UV can handle this automatically via `pyproject.toml`, but you can also set it manually:

```bash
export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
uv add /path/to/gdel3d_wgpu
```

## Verify Installation

```bash
# Activate UV environment
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate  # Windows

# Test import
python -c "import gdel3d; print(gdel3d.__version__)"

# Should output: 0.1.0
```

## Example: New Project Setup

Complete example of setting up a new project with gdel3d:

```bash
# Create new project
mkdir my_delaunay_project
cd my_delaunay_project

# Initialize with uv
uv init

# Add gdel3d (choose one method)
uv add /home/amai/gdel3d_wgpu                    # Local path
# uv add git+https://github.com/user/gdel3d.git  # Git URL

# Your pyproject.toml will now include gdel3d

# Create a test script
cat > test_delaunay.py << 'EOF'
import numpy as np
import gdel3d

points = np.random.rand(100, 3).astype(np.float32)
result = gdel3d.delaunay(points)
print(f"Generated {result.num_tets} tetrahedra")
EOF

# Run it
uv run test_delaunay.py
```

## Development Mode

If you're developing gdel3d itself:

```bash
cd /home/amai/gdel3d_wgpu

# Create UV environment
uv venv

# Install in editable mode with dev dependencies
uv pip install -e ".[dev]"

# Run tests
uv run pytest python/tests/ -v
```

## Troubleshooting

### "Build failed"

Make sure environment variable is set:
```bash
export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
uv add /path/to/gdel3d_wgpu
```

### "Rust compiler not found"

Install Rust:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

### "maturin not found"

UV will automatically install maturin as a build dependency. If issues persist:
```bash
uv pip install maturin
```

## UV Commands Quick Reference

```bash
# Add package from local path
uv add /path/to/gdel3d_wgpu

# Add package from git
uv add git+https://github.com/user/repo.git

# Install all dependencies
uv sync

# Run command in UV environment
uv run python script.py

# Run tests
uv run pytest

# Update dependencies
uv lock --upgrade

# Show installed packages
uv pip list
```

## Advantages of UV

- ⚡ **Fast**: 10-100x faster than pip
- 🔒 **Reproducible**: Lock files ensure consistent installs
- 🎯 **Modern**: Follows latest Python packaging standards
- 🛠️ **Built-in tooling**: No need for pip, virtualenv, etc.
- 📦 **Unified**: Manages Python versions and packages

## Example pyproject.toml for Your Project

```toml
[project]
name = "my-delaunay-app"
version = "0.1.0"
description = "My application using gdel3d"
requires-python = ">=3.9"
dependencies = [
    "numpy>=1.20",
    "gdel3d @ file:///home/amai/gdel3d_wgpu",  # Local path
    # or
    # "gdel3d @ git+https://github.com/user/gdel3d_wgpu.git",  # Git URL
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "ipython",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

## Next Steps

After installation:
- See [QUICKSTART_PYTHON.md](./QUICKSTART_PYTHON.md) for usage examples
- See [PYTHON_BINDINGS.md](./PYTHON_BINDINGS.md) for full API reference
- Run benchmark: `uv run python python/examples/benchmark.py`
