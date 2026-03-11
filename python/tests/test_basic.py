"""Basic tests for gdel3d Python bindings."""

import numpy as np
import pytest


def test_import():
    """Test that the module can be imported."""
    import gdel3d
    assert gdel3d.__version__ is not None


def test_simple_tetrahedron():
    """Test with 4 points forming a tetrahedron."""
    import gdel3d

    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)

    result = gdel3d.delaunay(points)

    assert result.num_tets >= 1, "Should have at least 1 tetrahedron"
    assert result.tets.shape[1] == 4, "Each tet should have 4 vertices"
    assert result.adjacency.shape == result.tets.shape, "Adjacency shape should match tets"


def test_cube():
    """Test with 8 points forming a cube."""
    import gdel3d

    points = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=np.float32)

    result = gdel3d.delaunay(points)

    assert result.num_tets >= 5, "Cube should have at least 5 tetrahedra"
    assert result.num_failed == 0, "All vertices should be inserted"


def test_random_points():
    """Test with random points."""
    import gdel3d

    np.random.seed(42)
    points = np.random.rand(100, 3).astype(np.float32)

    result = gdel3d.delaunay(points)

    assert result.num_tets > 0, "Should generate tetrahedra"
    assert result.tets.shape[1] == 4, "Each tet should have 4 vertices"


def test_config():
    """Test configuration options."""
    import gdel3d

    np.random.seed(42)
    points = np.random.rand(100, 3).astype(np.float32)

    config = gdel3d.Config(
        insertion_rule="circumcenter",
        enable_flipping=True,
        enable_splaying=True,
        max_insert_iterations=100,
        max_flip_iterations=10,
    )

    result = gdel3d.delaunay(points, config)
    assert result.num_tets > 0


def test_config_midpoint():
    """Test midpoint insertion rule."""
    import gdel3d

    np.random.seed(42)
    points = np.random.rand(50, 3).astype(np.float32)

    config = gdel3d.Config(insertion_rule="midpoint")
    result = gdel3d.delaunay(points, config)
    assert result.num_tets > 0


def test_config_invalid_rule():
    """Test invalid insertion rule."""
    import gdel3d

    with pytest.raises(ValueError, match="Invalid insertion_rule"):
        gdel3d.Config(insertion_rule="invalid")


def test_decode_encode_adjacency():
    """Test adjacency encoding/decoding."""
    import gdel3d

    tet_idx, face_idx = 42, 2
    packed = gdel3d.encode_adjacency(tet_idx, face_idx)
    decoded_tet, decoded_face = gdel3d.decode_adjacency(packed)

    assert decoded_tet == tet_idx
    assert decoded_face == face_idx


def test_invalid_constant():
    """Test INVALID constant."""
    import gdel3d

    assert gdel3d.INVALID == 2**32 - 1


def test_gpu_info():
    """Test GPU info functions."""
    import gdel3d

    info = gdel3d.gpu_info()
    assert isinstance(info, str)
    assert len(info) > 0


def test_initialize_gpu():
    """Test explicit GPU initialization."""
    import gdel3d

    info = gdel3d.initialize_gpu()
    assert isinstance(info, str)
    assert "GPU initialized" in info or "Limits" in info


def test_invalid_input_shape():
    """Test with invalid input shape."""
    import gdel3d

    # Wrong number of dimensions
    points = np.array([1, 2, 3], dtype=np.float32)
    with pytest.raises(ValueError, match="must have shape"):
        gdel3d.delaunay(points)

    # Wrong second dimension
    points = np.array([[1, 2], [3, 4]], dtype=np.float32)
    with pytest.raises(ValueError, match="must have shape"):
        gdel3d.delaunay(points)


def test_too_few_points():
    """Test with fewer than 4 points."""
    import gdel3d

    points = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
    with pytest.raises(ValueError, match="at least 4 points"):
        gdel3d.delaunay(points)


def test_result_repr():
    """Test DelaunayResult representation."""
    import gdel3d

    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)

    result = gdel3d.delaunay(points)
    repr_str = repr(result)

    assert "DelaunayResult" in repr_str
    assert "num_tets" in repr_str


def test_config_repr():
    """Test Config representation."""
    import gdel3d

    config = gdel3d.Config(
        insertion_rule="circumcenter",
        enable_flipping=True,
    )
    repr_str = repr(config)

    assert "Config" in repr_str
    assert "circumcenter" in repr_str
    assert "flipping=True" in repr_str


def test_large_dataset():
    """Test with a larger dataset (1000 points)."""
    import gdel3d

    np.random.seed(123)
    points = np.random.rand(1000, 3).astype(np.float32)

    result = gdel3d.delaunay(points)

    assert result.num_tets > 100, "Should generate many tetrahedra"
    assert result.tets.dtype == np.uint32
    assert result.adjacency.dtype == np.uint32
    assert result.failed_verts.dtype == np.uint32
