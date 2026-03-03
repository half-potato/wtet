// split_points.wgsl
// Port of kerSplitPointsFast from gDel3D/GDelFlipping/src/gDel3D/GPU/KerPredicates.cu:285-303
//
// Updates vert_tet for uninserted vertices whose containing tets are being split.
// Uses an orient3d decision tree to determine which of the 4 new split tets contains each vertex.

@group(0) @binding(0) var<storage, read> points: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> tets: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> tet_opp: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> tet_info: array<u32>;
@group(0) @binding(4) var<storage, read_write> vert_tet: array<u32>;
@group(0) @binding(5) var<storage, read> free_arr: array<u32>;
@group(0) @binding(6) var<storage, read> vert_free_arr: array<u32>;
@group(0) @binding(7) var<storage, read_write> counters: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read> uninserted: array<u32>;
@group(0) @binding(9) var<storage, read> tet_to_vert: array<u32>; // Maps tet idx -> position in uninserted array

const MEAN_VERTEX_DEGREE: u32 = 8u;
const INVALID: u32 = 0xFFFFFFFFu;

// SplitFaces[11][3] - Face vertices for orient3d decision tree
// From GPUDecl.h:167-172
const SPLIT_FACES: array<array<u32, 3>, 11> = array(
    array(0u, 1u, 4u),  // 0
    array(0u, 3u, 4u),  // 1
    array(0u, 2u, 4u),  // 2
    array(2u, 3u, 4u),  // 3
    array(1u, 3u, 4u),  // 4
    array(1u, 2u, 4u),  // 5
    array(2u, 3u, 4u),  // 6
    array(1u, 3u, 2u),  // 7
    array(0u, 2u, 3u),  // 8
    array(0u, 3u, 1u),  // 9
    array(0u, 1u, 2u),  // 10
);

// SplitNext[11][2] - Next face index based on orient3d result
// From GPUDecl.h:174-179
// [face][0] = next if orient positive, [face][1] = next if orient negative/zero
const SPLIT_NEXT: array<array<u32, 2>, 11> = array(
    array(1u, 2u),    // 0
    array(3u, 4u),    // 1
    array(5u, 6u),    // 2
    array(7u, 8u),    // 3
    array(9u, 7u),    // 4
    array(7u, 10u),   // 5
    array(7u, 8u),    // 6
    array(1u, 0u),    // 7
    array(2u, 0u),    // 8
    array(3u, 0u),    // 9
    array(4u, 0u),    // 10
);

// Fast orient3d test (uses only floating point, no exact arithmetic)
// Returns: > 0 if d is above plane abc, < 0 if below, ~0 if on plane
fn orient3d_fast(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, d: vec3<f32>) -> f32 {
    let ab = b - a;
    let ac = c - a;
    let ad = d - a;
    return dot(cross(ab, ac), ad);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let vert_idx = global_id.x;
    let num_uninserted = arrayLength(&uninserted);

    if (vert_idx >= num_uninserted) {
        return;
    }

    // Track how many threads enter
    atomicAdd(&counters[5], 1u); // scratch[1]

    // Get current tet for this uninserted vertex
    var tet_idx = vert_tet[vert_idx];

    if (tet_idx == INVALID) {
        return; // Vertex not yet located
    }

    // Track how many have valid tet_idx
    atomicAdd(&counters[4], 1u); // scratch[2]

    // Check if this tet is being split
    let split_vert_position = tet_to_vert[tet_idx];

    if (split_vert_position == INVALID) {
        return; // Tet not being split, nothing to update
    }

    // Get vertex ID of the vertex being inserted (that's splitting this tet)
    let split_vertex = uninserted[split_vert_position];

    // Get the actual vertex ID for this uninserted point
    let vertex = uninserted[vert_idx];

    // Load the original tet being split
    let tet = tets[tet_idx];

    // Build array of 5 vertices: 4 from original tet + 1 being inserted
    var tet_verts = array<u32, 5>(tet.x, tet.y, tet.z, tet.w, split_vertex);

    // Load points for all 5 vertices
    var pts = array<vec3<f32>, 5>(
        points[tet_verts[0]].xyz,
        points[tet_verts[1]].xyz,
        points[tet_verts[2]].xyz,
        points[tet_verts[3]].xyz,
        points[tet_verts[4]].xyz,
    );

    let pt_vertex = points[vertex].xyz;

    // Decision tree: 3 iterations to determine which of 4 new tets contains this vertex
    var face = 0u;

    for (var i = 0u; i < 3u; i++) {
        // Get face vertices for current face
        let fv = SPLIT_FACES[face];

        // Orient3d test: is vertex above or below this face?
        let orient = orient3d_fast(
            pts[fv[0]],
            pts[fv[1]],
            pts[fv[2]],
            pt_vertex
        );

        // Navigate to next face based on orientation
        if (orient > 0.0) {
            face = SPLIT_NEXT[face][0]; // Positive orientation
        } else {
            face = SPLIT_NEXT[face][1]; // Negative or zero orientation
        }
    }

    // After 3 iterations, face should be in range 7-10
    // Map to actual tet index via free_arr
    // The split operation allocates 4 tets from the split vertex's free list
    // They are at positions: free_idx, free_idx-1, free_idx-2, free_idx-3
    // where free_idx = (split_vertex + 1) * MEAN_VERTEX_DEGREE - 1 (top of stack)

    if (face >= 7u && face <= 10u) {
        let free_idx = (split_vertex + 1u) * MEAN_VERTEX_DEGREE - 1u;
        let offset = face - 7u; // 0, 1, 2, or 3

        // Bounds check
        if (free_idx >= offset && free_idx - offset < arrayLength(&free_arr)) {
            let new_tet = free_arr[free_idx - offset];
            vert_tet[vert_idx] = new_tet;
            // Increment counter to track updates
            atomicAdd(&counters[6], 1u); // Use scratch[0] counter
        }
    }
}
