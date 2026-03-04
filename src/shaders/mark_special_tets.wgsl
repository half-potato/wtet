// Kernel: Mark Special Tets
//
// Builds a queue of tets that have OPP_SPECIAL flags (set by fast phase when
// predicates were uncertain). These tets need exact predicate checking.
// Outputs compacted queue to act_tet_vec and count to counters[0].
//
// Two-phase approach:
// 1. Scan all alive tets for OPP_SPECIAL flags
// 2. Atomically allocate slots in act_tet_vec for special tets
// 3. Clear special flags after queuing
//
// Port of kerMarkSpecialTets from KerDivision.cu:834 (modified for queue building)

@group(0) @binding(0) var<storage, read_write> tet_info: array<u32>;
@group(0) @binding(1) var<storage, read_write> tet_opp: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> act_tet_vec: array<i32>;
@group(0) @binding(3) var<storage, read_write> counters: array<atomic<u32>>;
@group(0) @binding(4) var<uniform> params: vec4<u32>; // x = tet_num

const TET_ALIVE: u32 = 1u;
const TET_CHANGED: u32 = 2u;
const OPP_SPECIAL: u32 = 8u; // Bit 3

fn is_tet_alive(info: u32) -> bool {
    return (info & TET_ALIVE) != 0u;
}

fn is_opp_special(opp: u32) -> bool {
    return (opp & OPP_SPECIAL) != 0u;
}

fn clear_opp_special(opp: u32) -> u32 {
    return opp & ~OPP_SPECIAL;
}

@compute @workgroup_size(64)
fn mark_special_tets(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let tet_num = params.x;

    // Reset counter on first thread
    if gid.x == 0u {
        atomicStore(&counters[0], 0u);
    }
    workgroupBarrier();

    if idx >= tet_num {
        return;
    }

    // Skip dead tets
    if !is_tet_alive(tet_info[idx]) {
        return;
    }

    // Load all 4 opp values
    var opp = array<u32, 4>(
        atomicLoad(&tet_opp[idx * 4u + 0u]),
        atomicLoad(&tet_opp[idx * 4u + 1u]),
        atomicLoad(&tet_opp[idx * 4u + 2u]),
        atomicLoad(&tet_opp[idx * 4u + 3u]),
    );

    var has_special = false;

    // Check each face for special bit
    for (var vi = 0u; vi < 4u; vi++) {
        if is_opp_special(opp[vi]) {
            has_special = true;
            opp[vi] = clear_opp_special(opp[vi]);
        }
    }

    // If this tet has special flags, add it to the queue for exact processing
    if has_special {
        // Atomically allocate a slot in act_tet_vec
        let slot = atomicAdd(&counters[0], 1u);
        act_tet_vec[slot] = i32(idx);

        // Mark tet as Changed and write back cleared opp values
        tet_info[idx] |= TET_CHANGED;
        atomicStore(&tet_opp[idx * 4u + 0u], opp[0]);
        atomicStore(&tet_opp[idx * 4u + 1u], opp[1]);
        atomicStore(&tet_opp[idx * 4u + 2u], opp[2]);
        atomicStore(&tet_opp[idx * 4u + 3u], opp[3]);
    }
}
