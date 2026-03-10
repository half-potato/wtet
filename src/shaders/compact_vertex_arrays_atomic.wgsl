// Atomic-based compaction for vertex arrays (fast path for small datasets)
// Two-pass algorithm:
//   Pass 0: Count non-inserted positions
//   Pass 1: Scatter non-inserted vertices to compacted arrays

@group(0) @binding(0) var<storage, read> insert_list: array<vec2<u32>>;  // (tet_idx, position)
@group(0) @binding(1) var<storage, read> uninserted_in: array<u32>;
@group(0) @binding(2) var<storage, read> vert_tet_in: array<u32>;
@group(0) @binding(3) var<storage, read_write> uninserted_out: array<u32>;
@group(0) @binding(4) var<storage, read_write> vert_tet_out: array<u32>;
@group(0) @binding(5) var<storage, read_write> counters: array<atomic<u32>>;
@group(0) @binding(6) var<uniform> params: vec4<u32>;  // x=num_uninserted, y=num_inserted, z=pass, w=unused

@compute @workgroup_size(256)
fn compact_atomic(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let idx = gid.x;
    let num_uninserted = params.x;
    let num_inserted = params.y;
    let pass_num = params.z;

    if idx >= num_uninserted {
        return;
    }

    // Check if this position was inserted
    var is_inserted = false;
    for (var i = 0u; i < num_inserted; i++) {
        if insert_list[i].y == idx {
            is_inserted = true;
            break;
        }
    }

    if pass_num == 0u {
        // Pass 0: Count non-inserted positions
        if !is_inserted {
            atomicAdd(&counters[0], 1u);
        }
    } else {
        // Pass 1: Scatter non-inserted vertices
        if !is_inserted {
            let out_idx = atomicAdd(&counters[1], 1u);
            uninserted_out[out_idx] = uninserted_in[idx];
            vert_tet_out[out_idx] = vert_tet_in[idx];
        }
    }
}
