// kerCollectFreeSlots - Collect indices of dead tets into free_arr
// Port from gDel3D/GDelFlipping/src/gDel3D/GPU/KerDivision.cu lines 1152-1168
//
// For each dead tet (idx < newTetNum):
//   freeIdx = idx - prefixArr[idx]
//   freeArr[freeIdx] = idx
//
// This builds an array of free slot indices that can be reused.

@group(0) @binding(0) var<storage, read> tet_info: array<u32>;
@group(0) @binding(1) var<storage, read> prefix_arr: array<u32>;
@group(0) @binding(2) var<storage, read_write> free_arr: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>; // x = new_tet_num

const TET_ALIVE: u32 = 1u;

fn is_tet_alive(info: u32) -> bool {
    return (info & TET_ALIVE) != 0u;
}

@compute @workgroup_size(64)
fn collect_free_slots(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let new_tet_num = params.x;

    if idx >= new_tet_num {
        return;
    }

    // Skip alive tets - we only collect dead slots
    if is_tet_alive(tet_info[idx]) {
        return;
    }

    // Calculate position in free array
    let free_idx = idx - prefix_arr[idx];

    // Store this dead tet's index
    free_arr[free_idx] = idx;
}
