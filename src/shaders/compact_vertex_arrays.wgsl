// GPU compaction for vertex arrays (uninserted + vert_tet) - FORCE RECOMPILE v2
// Removes vertices that were successfully inserted
// Port of thrust::remove_if for zip iterators (ThrustWrapper.cu:244-265)
//
// Two-pass algorithm:
// Pass 1: Count vertices NOT in insert_list (linear search on positions)
// Pass 2: Copy non-inserted vertices to compacted output

@group(0) @binding(0) var<storage, read> uninserted_in: array<u32>;
@group(0) @binding(1) var<storage, read> vert_tet_in: array<u32>;
@group(0) @binding(2) var<storage, read> insert_list: array<vec2<u32>>; // (tet_idx, position)
@group(0) @binding(3) var<storage, read_write> uninserted_out: array<u32>;
@group(0) @binding(4) var<storage, read_write> vert_tet_out: array<u32>;
@group(0) @binding(5) var<storage, read_write> counter: array<atomic<u32>>;
@group(0) @binding(6) var<uniform> params: vec4<u32>;
// x = num_uninserted (input size)
// y = num_inserted (insert_list size)
// z = pass_num (0 or 1)

// ┌────────────────────────────────────────────────────────────────────────┐
// │ CRITICAL: insert_list[i].y is POSITION (idx), NOT vertex ID!          │
// │ This matches CUDA kerPickWinnerPoint (KerDivision.cu:311-335)         │
// │ DO NOT CHANGE TO vert_idx - that breaks compaction logic!             │
// └────────────────────────────────────────────────────────────────────────┘
// Check if a position exists in insert_list (linear search - small list)
fn is_inserted(idx: u32, num_inserted: u32) -> bool {
    // Unroll loop to avoid any potential WGSL optimization issues
    if num_inserted >= 1u && insert_list[0].y == idx { return true; }
    if num_inserted >= 2u && insert_list[1].y == idx { return true; }
    if num_inserted >= 3u && insert_list[2].y == idx { return true; }
    if num_inserted >= 4u && insert_list[3].y == idx { return true; }
    if num_inserted >= 5u && insert_list[4].y == idx { return true; }
    if num_inserted >= 6u && insert_list[5].y == idx { return true; }
    if num_inserted >= 7u && insert_list[6].y == idx { return true; }
    if num_inserted >= 8u && insert_list[7].y == idx { return true; }
    return false;
}

@compute @workgroup_size(256)
fn compact_vertex_arrays(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let idx = gid.x;
    let num_uninserted = params.x;
    let num_inserted = params.y;
    let pass_num = params.z;

    if idx >= num_uninserted {
        return;
    }

    // Read input arrays (needed by both passes)
    let vert_idx = uninserted_in[idx];
    let tet_idx = vert_tet_in[idx];

    // Check if this position was inserted
    // NOTE: insert_list stores POSITION (idx), not vertex ID!
    let inserted = is_inserted(idx, num_inserted);

    if pass_num == 0u {
        // Pass 1: Count vertices that were NOT inserted
        if !inserted {
            atomicAdd(&counter[0], 1u);
        }
    } else {
        // Pass 2: Compact non-inserted vertices to output arrays
        if !inserted {
            let out_idx = atomicAdd(&counter[1], 1u);
            uninserted_out[out_idx] = vert_idx;
            vert_tet_out[out_idx] = tet_idx;
        }
    }
}
