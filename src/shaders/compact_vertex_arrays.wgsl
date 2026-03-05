// GPU compaction for vertex arrays (uninserted + vert_tet)
// Removes vertices that were successfully inserted
// Port of thrust::remove_if for zip iterators (ThrustWrapper.cu:244-265)
//
// Two-pass algorithm:
// Pass 1: Count vertices NOT in insert_list (using binary search)
// Pass 2: Copy non-inserted vertices to compacted output

@group(0) @binding(0) var<storage, read> uninserted_in: array<u32>;
@group(0) @binding(1) var<storage, read> vert_tet_in: array<u32>;
@group(0) @binding(2) var<storage, read> insert_list: array<vec2<u32>>; // (tet_idx, vert_idx)
@group(0) @binding(3) var<storage, read_write> uninserted_out: array<u32>;
@group(0) @binding(4) var<storage, read_write> vert_tet_out: array<u32>;
@group(0) @binding(5) var<storage, read_write> counter: array<atomic<u32>>;
@group(0) @binding(6) var<uniform> params: vec4<u32>;
// x = num_uninserted (input size)
// y = num_inserted (insert_list size)
// z = pass_num (0 or 1)

// Check if a vertex ID exists in insert_list (linear search - small list)
fn is_inserted(vert_idx: u32, num_inserted: u32) -> bool {
    for (var i = 0u; i < num_inserted; i++) {
        if insert_list[i].y == vert_idx {
            return true;
        }
    }
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

    let vert_idx = uninserted_in[idx];
    let tet_idx = vert_tet_in[idx];

    // Check if this vertex was inserted
    let inserted = is_inserted(vert_idx, num_inserted);

    if pass_num == 0u {
        // Pass 1: Count vertices that were NOT inserted
        if !inserted {
            atomicAdd(&counter[0], 1u);
        }
    } else {
        // Pass 2: Compact non-inserted vertices
        if !inserted {
            let out_idx = atomicAdd(&counter[1], 1u);
            uninserted_out[out_idx] = vert_idx;
            vert_tet_out[out_idx] = tet_idx;
        }
    }
}
