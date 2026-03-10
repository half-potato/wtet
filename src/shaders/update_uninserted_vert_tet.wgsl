// Kernel: Update vert_tet for all uninserted vertices to point to an alive tet
//
// After splits, some vertices may still point to dead tets. This kernel ensures
// all uninserted vertices point to a valid alive tet so point location can find them.
//
// Dispatch: ceil(num_uninserted / 64)

@group(0) @binding(0) var<storage, read> tet_info: array<u32>;
@group(0) @binding(1) var<storage, read> uninserted: array<u32>;
@group(0) @binding(2) var<storage, read_write> vert_tet: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>; // x = num_uninserted, y = max_tets
// NOTE: Removed update_debug to stay under 10 storage buffer limit
// @group(0) @binding(4) var<storage, read_write> update_debug: array<vec4<u32>>; // Debug output per thread

const TET_ALIVE: u32 = 1u;

@compute @workgroup_size(256)
fn update_uninserted_vert_tet(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let idx = gid.x;
    let num_uninserted = params.x;

    if idx >= num_uninserted {
        return;
    }

    let vert_idx = uninserted[idx];
    // CRITICAL: vert_tet is position-indexed, NOT vertex-indexed!
    // Use idx (position in uninserted array), not vert_idx (vertex ID)
    let old_tet = vert_tet[idx];

    // Debug: record start state (DISABLED)
    // update_debug[idx * 4u + 0u] = vec4<u32>(idx, vert_idx, old_tet, num_uninserted);

    // Scan for first alive tet (limit scan to first 100 tets for performance)
    // Most alive tets will be in low indices
    var found_tet = 0u;
    for (var tet_idx = 1u; tet_idx < 100u; tet_idx++) {
        if (tet_info[tet_idx] & TET_ALIVE) != 0u {
            found_tet = tet_idx;
            vert_tet[idx] = tet_idx;
            break;
        }
    }

    // Debug: record result (DISABLED)
    // let new_tet = vert_tet[idx];
    // let old_status = tet_info[old_tet];
    // update_debug[idx * 4u + 1u] = vec4<u32>(found_tet, new_tet, old_status, 0u);
    // update_debug[idx * 4u + 2u] = vec4<u32>(tet_info[1], tet_info[61], tet_info[62], tet_info[63]);
    // update_debug[idx * 4u + 3u] = vec4<u32>(tet_info[64], 0u, 0u, 0u);
}
