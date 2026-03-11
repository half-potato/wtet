"""Basic example of using gdel3d for 3D Delaunay triangulation."""

import numpy as np
import gdel3d

# Initialize GPU (optional, auto-called on first use)
print("GPU Info:")
print(gdel3d.gpu_info())
print()

# Generate random 3D points
np.random.seed(42)
num_points = 1000
points = np.random.rand(num_points, 3).astype(np.float32)

print(f"Input: {points.shape[0]} random points in [0, 1]³")
print()

# Create configuration
config = gdel3d.Config(
    insertion_rule="circumcenter",
    enable_flipping=True,
    enable_splaying=True,
    max_insert_iterations=100,
    max_flip_iterations=10,
)

print(f"Configuration: {config}")
print()

# Compute Delaunay triangulation
print("Computing Delaunay triangulation...")
result = gdel3d.delaunay(points, config)

print(f"\nResults:")
print(f"  Generated: {result.num_tets} tetrahedra")
print(f"  Failed to insert: {result.num_failed} vertices")
print()

# Access result arrays
print(f"Result arrays:")
print(f"  tets shape: {result.tets.shape} (dtype: {result.tets.dtype})")
print(f"  adjacency shape: {result.adjacency.shape} (dtype: {result.adjacency.dtype})")
print(f"  failed_verts shape: {result.failed_verts.shape} (dtype: {result.failed_verts.dtype})")
print()

# Show first few tetrahedra
print("First 5 tetrahedra:")
for i in range(min(5, result.num_tets)):
    tet = result.tets[i]
    print(f"  Tet {i}: vertices {tet}")

    # Decode adjacency for each face
    for face in range(4):
        adj = result.adjacency[i, face]
        if adj != gdel3d.INVALID:
            neighbor_tet, neighbor_face = gdel3d.decode_adjacency(adj)
            print(f"    Face {face} → Tet {neighbor_tet}, Face {neighbor_face}")
        else:
            print(f"    Face {face} → boundary (no neighbor)")

print()

# Show encoding/decoding
print("Adjacency encoding example:")
tet_idx, face_idx = 123, 2
encoded = gdel3d.encode_adjacency(tet_idx, face_idx)
decoded_tet, decoded_face = gdel3d.decode_adjacency(encoded)
print(f"  encode({tet_idx}, {face_idx}) = {encoded}")
print(f"  decode({encoded}) = ({decoded_tet}, {decoded_face})")
print()

print("Done!")
