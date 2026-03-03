// Kernel: Shift opponent tet indices
//
// Shifts tet indices in the opponent array that are >= start by adding shift.
// This is used during tet array reorganization.
//
// Corresponds to kerShiftOppTetIdx in KerDivision.cu line 1078
//
// Dispatch: ceil(tet_num / 64)

@group(0) @binding(0) var<storage, read_write> tet_opp: array<atomic<u32>>;
@group(0) @binding(1) var<uniform> params: vec4<u32>; // x = tet_num, y = start, z = shift

const INVALID: u32 = 0xFFFFFFFFu;

// TetOpp encoding: (tet_idx << 5u) | opp_vi (original CUDA encoding with flag bits)
// Bits 0-1: vi, Bit 2: internal, Bit 3: special, Bit 4: sphere_fail, Bits 5-31: tet_idx
fn decode_opp_tet(packed: u32) -> u32 {
    return packed >> 5u;
}

fn decode_opp_vi(packed: u32) -> u32 {
    return packed & 3u;
}

fn encode_opp(tet_idx: u32, opp_vi: u32) -> u32 {
    return (tet_idx << 5u) | (opp_vi & 3u);
}

@compute @workgroup_size(64)
fn shift_opp_tet_idx(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let tet_num = params.x;
    let start = params.y;
    let shift = params.z;

    let idx = gid.x;
    if idx >= tet_num {
        return;
    }

    // Process all 4 faces of this tet
    for (var i = 0u; i < 4u; i = i + 1u) {
        let opp_idx_base = idx * 4u + i;
        let opp_packed = atomicLoad(&tet_opp[opp_idx_base]);

        // Skip invalid entries
        if opp_packed == INVALID {
            continue;
        }

        let opp_tet_idx = decode_opp_tet(opp_packed);
        let opp_vi = decode_opp_vi(opp_packed);

        // Only shift if >= start
        if opp_tet_idx >= start {
            atomicStore(&tet_opp[opp_idx_base], encode_opp(opp_tet_idx + shift, opp_vi));
        }
    }
}
