"""
GPU-accelerated 3D Delaunay triangulation using wgpu.

This package provides fast 3D Delaunay triangulation on the GPU using
wgpu compute shaders. It accepts NumPy arrays as input and returns
tetrahedra, adjacency information, and failed vertices.

Example:
    >>> import numpy as np
    >>> import wtet
    >>>
    >>> # Generate random 3D points
    >>> points = np.random.rand(1000, 3).astype(np.float32)
    >>>
    >>> # Compute Delaunay triangulation
    >>> result = wtet.delaunay(points)
    >>>
    >>> print(f"Generated {result.num_tets} tetrahedra")
    >>> print(f"Failed to insert {result.num_failed} vertices")
    >>>
    >>> # Access tetrahedra (N, 4) array of vertex indices
    >>> tets = result.tets
    >>>
    >>> # Access adjacency (N, 4) array of packed (tet_idx, face_idx)
    >>> adjacency = result.adjacency
    >>>
    >>> # Decode adjacency
    >>> for i, tet in enumerate(tets[:5]):
    ...     print(f"Tet {i}: vertices {tet}")
    ...     for face in range(4):
    ...         adj = adjacency[i, face]
    ...         if adj != wtet.INVALID:
    ...             neighbor_tet, neighbor_face = wtet.decode_adjacency(adj)
    ...             print(f"  Face {face} → Tet {neighbor_tet}, Face {neighbor_face}")

Configuration:
    >>> config = wtet.Config(
    ...     insertion_rule="circumcenter",  # or "centroid"
    ...     enable_flipping=True,
    ...     enable_splaying=True,
    ...     max_insert_iterations=100,
    ...     max_flip_iterations=10,
    ... )
    >>> result = wtet.delaunay(points, config)
"""

from ._core import (
    delaunay,
    Config,
    DelaunayResult,
    decode_adjacency,
    encode_adjacency,
    initialize_gpu,
    gpu_info,
    INVALID,
    __version__,
)

__all__ = [
    "delaunay",
    "Config",
    "DelaunayResult",
    "decode_adjacency",
    "encode_adjacency",
    "initialize_gpu",
    "gpu_info",
    "INVALID",
    "__version__",
]
