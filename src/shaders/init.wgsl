// Kernel: Create the initial super-tetrahedron that contains all input points.
//
// The super-tetrahedron uses 4 virtual vertices placed far enough to contain
// the entire [0,1]³ bounding box. Vertex index N is the "infinity" point.
//
// Dispatch: (1, 1, 1) — single invocation.

@group(0) @binding(0) var<storage, read> points: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> tets: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> tet_opp: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> tet_info: array<u32>;
@group(0) @binding(4) var<storage, read_write> vert_tet: array<u32>;
@group(0) @binding(5) var<storage, read_write> free_arr: array<u32>; // unused for now
@group(0) @binding(6) var<storage, read_write> vert_free_arr: array<u32>; // unused for now
@group(0) @binding(7) var<storage, read_write> counters: array<atomic<u32>>;
@group(0) @binding(8) var<uniform> params: vec4<u32>; // x = num_points, y = max_tets

const INVALID: u32 = 0xFFFFFFFFu;
const TET_ALIVE: u32 = 1u;

// Super-tetrahedron virtual vertex indices.
// These are placed at indices [num_points .. num_points+3].
// The actual coordinates for them are set on the CPU side before upload.

@compute @workgroup_size(1)
fn make_first_tetra() {
    let n = params.x;

    // Virtual vertices: n, n+1, n+2, n+3
    // Form a single super-tetrahedron with positive orientation.
    let v0 = n;      // super vertex 0
    let v1 = n + 1u; // super vertex 1
    let v2 = n + 2u; // super vertex 2
    let v3 = n + 3u; // super vertex 3

    // Create one tet with positive orientation (orient3d > 0).
    // The raw order (v0,v1,v2,v3) gives negative orient3d, so swap v2↔v3.
    tets[0] = vec4<u32>(v0, v1, v3, v2);
    atomicStore(&tet_opp[0u], INVALID);
    atomicStore(&tet_opp[1u], INVALID);
    atomicStore(&tet_opp[2u], INVALID);
    atomicStore(&tet_opp[3u], INVALID);
    tet_info[0] = TET_ALIVE;

    // All input points start in tet 0
    for (var i = 0u; i < n; i++) {
        vert_tet[i] = 0u;
    }

    // Set counters (free stack was pre-filled by CPU with max_tets-1 entries)
    let max_tets = params.y;
    atomicStore(&counters[0], max_tets - 1u); // free_count = max_tets - 1 (tet 0 is used)
    atomicStore(&counters[1], 1u);            // active_count = 1
    atomicStore(&counters[2], 0u);            // inserted_count = 0
    atomicStore(&counters[3], 0u);            // failed_count = 0
}
