// kerMakeCompactMap - Build compact map for alive tets beyond newTetNum
// Port from gDel3D/GDelFlipping/src/gDel3D/GPU/KerDivision.cu lines 1171-1188
//
// For each alive tet (idx >= newTetNum):
//   freeIdx = newTetNum - prefixArr[idx]
//   newTetIdx = freeArr[freeIdx]
//   prefixArr[idx] = newTetIdx  (reuse prefix_arr as compact map)
//
// This assigns new indices to alive tets that need to be moved down.

@group(0) @binding(0) var<storage, read> tet_info: array<u32>;
@group(0) @binding(1) var<storage, read_write> prefix_arr: array<u32>;
@group(0) @binding(2) var<storage, read> free_arr: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>; // x = new_tet_num, y = tet_info_num

const TET_ALIVE: u32 = 1u;

fn is_tet_alive(info: u32) -> bool {
    return (info & TET_ALIVE) != 0u;
}

@compute @workgroup_size(64)
fn make_compact_map(@builtin(global_invocation_id) gid: vec3<u32>) {
    let new_tet_num = params.x;
    let tet_info_num = params.y;
    let idx = new_tet_num + gid.x;

    if idx >= tet_info_num {
        return;
    }

    // Skip dead tets - only alive tets get new indices
    if !is_tet_alive(tet_info[idx]) {
        return;
    }

    // Calculate where this tet should go
    // Note: prefix_arr[idx] should always be <= new_tet_num for alive tets
    let prefix_count = prefix_arr[idx];

    // Bounds check to prevent underflow
    if (prefix_count > new_tet_num) {
        // ERROR: This should never happen!
        // prefix_count > new_tet_num means more alive tets than total count
        prefix_arr[idx] = 0u;  // Map to first slot as fallback
        return;
    }

    let free_idx = new_tet_num - prefix_count;
    let new_tet_idx = free_arr[free_idx];

    // Store new index in prefix_arr (repurposing it as compact map)
    prefix_arr[idx] = new_tet_idx;
}
