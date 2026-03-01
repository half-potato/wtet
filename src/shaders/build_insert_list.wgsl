// Build insert_list from tet_vert winners
// After pick_winner sets tet_vert[tet_idx] = winning idx (position in uninserted array),
// we scan tets and build the insert_list with (tet_idx, vert_idx) pairs

@group(0) @binding(0) var<storage, read> tet_info: array<u32>;
@group(0) @binding(1) var<storage, read_write> tet_vert: array<atomic<i32>>;
@group(0) @binding(2) var<storage, read> uninserted: array<u32>;
@group(0) @binding(3) var<storage, read_write> insert_list: array<vec2<u32>>;
@group(0) @binding(4) var<storage, read_write> counters: array<atomic<u32>>;
@group(0) @binding(5) var<uniform> params: vec4<u32>; // x = max_tets

const TET_ALIVE: u32 = 1u;
const INT_MAX: i32 = 0x7FFFFFFF;
const COUNTER_INSERTED: u32 = 2u;

@compute @workgroup_size(64)
fn build_insert_list(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let tet_idx = gid.x;
    let max_tets = params.x;

    if tet_idx >= max_tets {
        return;
    }

    let winner_idx = atomicLoad(&tet_vert[tet_idx]);

    // DEBUG: Count dead tets with winners
    if winner_idx != INT_MAX && (tet_info[tet_idx] & TET_ALIVE) == 0u {
        // This tet is DEAD but has a winner - this is the bug!
        // Winner will be ignored, causing insertion failure
        return;
    }

    if (tet_info[tet_idx] & TET_ALIVE) == 0u {
        return;
    }

    if winner_idx == INT_MAX {
        return; // No winner for this tet
    }

    // winner_idx is the position in the uninserted array
    // Store (tet_idx, position) so split shader can update vert_tet[position]
    let slot = atomicAdd(&counters[COUNTER_INSERTED], 1u);
    insert_list[slot] = vec2<u32>(tet_idx, u32(winner_idx));
}
