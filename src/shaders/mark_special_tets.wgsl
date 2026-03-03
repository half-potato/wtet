// Kernel: Mark Special Tets
//
// Clears the "special" bit on all opp entries for alive tets.
// The "special" bit (bit 3) is set during Delaunay checking when exact predicates
// are needed. This kernel resets those flags and marks tets as "Changed" if any
// special bits were cleared.
//
// Port of kerMarkSpecialTets from KerDivision.cu:834

@group(0) @binding(0) var<storage, read_write> tet_info: array<u32>;
@group(0) @binding(1) var<storage, read_write> tet_opp: array<atomic<u32>>;
@group(0) @binding(2) var<uniform> params: vec4<u32>; // x = tet_num

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

    var changed = false;

    // Check each face for special bit
    for (var vi = 0u; vi < 4u; vi++) {
        if is_opp_special(opp[vi]) {
            changed = true;
            opp[vi] = clear_opp_special(opp[vi]);
        }
    }

    // If any special bits were cleared, mark tet as Changed and write back opp
    if changed {
        tet_info[idx] |= TET_CHANGED;
        atomicStore(&tet_opp[idx * 4u + 0u], opp[0]);
        atomicStore(&tet_opp[idx * 4u + 1u], opp[1]);
        atomicStore(&tet_opp[idx * 4u + 2u], opp[2]);
        atomicStore(&tet_opp[idx * 4u + 3u], opp[3]);
    }
}
