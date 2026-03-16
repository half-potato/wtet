// Collect indices of alive tets into act_tet_vec.
// Equivalent to CUDA's thrust_copyIf_IsActiveTetra (ThrustWrapper.cu:267-289).
//
// Uses atomic counter for stream compaction (order doesn't matter for flip checking).
//
// params.y controls filtering:
//   0 = collect ALIVE & CHANGED tets only (default, for incremental collection)
//   1 = collect ALL alive tets (for initial flip queue population)
//
// Bindings:
// - @binding(0): tet_info (read) - per-tet flags
// - @binding(1): act_tet_vec (read_write) - output: indices of active tets
// - @binding(2): counters (read_write) - counters[0] used as atomic output count
// - @binding(3): params (uniform) - x = total_tet_num, y = collect_all (0 or 1)

@group(0) @binding(0) var<storage, read> tet_info: array<u32>;
@group(0) @binding(1) var<storage, read_write> act_tet_vec: array<i32>;
@group(0) @binding(2) var<storage, read_write> counters: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

const TET_ALIVE: u32 = 1u;
const TET_CHANGED: u32 = 2u;

@compute @workgroup_size(256)
fn collect_active_tets(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.x {
        return;
    }

    let info = tet_info[idx];
    let is_alive = (info & TET_ALIVE) != 0u;
    let is_changed = (info & TET_CHANGED) != 0u;
    let collect_all = params.y != 0u;

    if is_alive && (collect_all || is_changed) {
        let slot = atomicAdd(&counters[0], 1u);
        act_tet_vec[slot] = i32(idx);
    }
}
