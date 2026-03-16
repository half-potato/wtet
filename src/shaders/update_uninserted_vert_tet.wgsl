// Kernel: Update vert_tet for all uninserted vertices to point to an alive tet
//
// After flips, some vertices may point to dead tets (destroyed by 3-2 flips).
// This kernel walks adjacency from dead tets to find alive neighbors.
//
// Strategy:
// 1. If old_tet is alive → keep it (fast path, most common case)
// 2. If dead, check its 4 neighbors via tet_opp → use first alive neighbor
// 3. If no alive neighbor, scan tet_info for any alive tet (fallback)
//
// Dispatch: ceil(num_uninserted / 256)

@group(0) @binding(0) var<storage, read> tet_info: array<u32>;
@group(0) @binding(1) var<storage, read> uninserted: array<u32>;
@group(0) @binding(2) var<storage, read_write> vert_tet: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>; // x = num_uninserted, y = max_tets
@group(0) @binding(4) var<storage, read> tet_opp: array<u32>;

const TET_ALIVE: u32 = 1u;

@compute @workgroup_size(256)
fn update_uninserted_vert_tet(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let idx = gid.x;
    let num_uninserted = params.x;
    let max_tets = params.y;

    if idx >= num_uninserted {
        return;
    }

    // CRITICAL: vert_tet is position-indexed, NOT vertex-indexed!
    // Use idx (position in uninserted array), not vert_idx (vertex ID)
    let old_tet = vert_tet[idx];

    // Fast path: old_tet is still alive → nothing to repair
    if old_tet < max_tets && (tet_info[old_tet] & TET_ALIVE) != 0u {
        return;
    }

    // Old tet is dead. Walk adjacency to find alive neighbor.
    // tet_opp uses 5-bit encoding: tet_idx = packed >> 5u
    if old_tet < max_tets {
        for (var face = 0u; face < 4u; face++) {
            let packed = tet_opp[old_tet * 4u + face];
            let nei_tet = packed >> 5u;
            if nei_tet < max_tets && (tet_info[nei_tet] & TET_ALIVE) != 0u {
                vert_tet[idx] = nei_tet;
                return;
            }
        }

        // Second ring: check neighbors of neighbors
        for (var face = 0u; face < 4u; face++) {
            let packed = tet_opp[old_tet * 4u + face];
            let nei_tet = packed >> 5u;
            if nei_tet < max_tets {
                for (var face2 = 0u; face2 < 4u; face2++) {
                    let packed2 = tet_opp[nei_tet * 4u + face2];
                    let nei2_tet = packed2 >> 5u;
                    if nei2_tet < max_tets && (tet_info[nei2_tet] & TET_ALIVE) != 0u {
                        vert_tet[idx] = nei2_tet;
                        return;
                    }
                }
            }
        }
    }

    // Fallback: scan for ANY alive tet
    for (var t = 0u; t < max_tets; t++) {
        if (tet_info[t] & TET_ALIVE) != 0u {
            vert_tet[idx] = t;
            return;
        }
    }
}
