// Kernels: Vote for which point to insert into each tetrahedron.
//
// Two-pass approach:
//   1. kerVoteForPoint: Each uninserted point votes for its containing tet.
//      Uses atomicMax to pick the point furthest outside circumsphere (best quality).
//   2. kerPickWinnerPoint: Each tet that received a vote marks the winner.
//
// Port of kerVoteForPoint from KerPredicates.cu:160-200
// Uses inSphereDet (signed distance to circumsphere) for voting.

@group(0) @binding(0) var<storage, read> points: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> tets: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> tet_info: array<u32>;
@group(0) @binding(3) var<storage, read_write> tet_vote: array<atomic<i32>>;
@group(0) @binding(4) var<storage, read_write> vert_sphere: array<i32>;
@group(0) @binding(5) var<storage, read> vert_tet: array<u32>;
@group(0) @binding(6) var<storage, read> uninserted: array<u32>;
@group(0) @binding(7) var<uniform> params: vec4<u32>; // x = num_uninserted

const INVALID: u32 = 0xFFFFFFFFu;
const NO_VOTE: i32 = i32(0x80000000u);  // Minimum i32 for atomicMax voting
const TET_ALIVE: u32 = 1u;

// Fast insphere: returns (det, uncertain) where uncertain=1.0 if error bounds exceeded
fn insphere_fast(
    ax: f32, ay: f32, az: f32,
    bx: f32, by: f32, bz: f32,
    cx: f32, cy: f32, cz: f32,
    dx: f32, dy: f32, dz: f32,
    ex: f32, ey: f32, ez: f32,
) -> vec2<f32> {
    let aex = ax - ex; let aey = ay - ey; let aez = az - ez;
    let bex = bx - ex; let bey = by - ey; let bez = bz - ez;
    let cex = cx - ex; let cey = cy - ey; let cez = cz - ez;
    let dex = dx - ex; let dey = dy - ey; let dez = dz - ez;

    let ab = aex * bey - bex * aey;
    let bc = bex * cey - cex * bey;
    let cd = cex * dey - dex * cey;
    let da = dex * aey - aex * dey;
    let ac = aex * cey - cex * aey;
    let bd = bex * dey - dex * bey;

    let abc = aez * bc - bez * ac + cez * ab;
    let bcd = bez * cd - cez * bd + dez * bc;
    let cda = cez * da + dez * ac + aez * cd;
    let dab = dez * ab + aez * bd + bez * da;

    let alift = aex * aex + aey * aey + aez * aez;
    let blift = bex * bex + bey * bey + bez * bez;
    let clift = cex * cex + cey * cey + cez * cez;
    let dlift = dex * dex + dey * dey + dez * dez;

    let det = (dlift * abc - clift * dab) + (blift * cda - alift * bcd);

    // For voting, we don't need full error bound checking - just use the fast result
    // CUDA uses fast float predicates for voting (not exact)
    return vec2<f32>(det, 0.0);
}

@compute @workgroup_size(64)
fn vote_for_point(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let idx = gid.x;
    let num_uninserted = params.x;

    if idx >= num_uninserted {
        return;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // ⚠️  CRITICAL: vert_tet is POSITION-INDEXED, NOT VERTEX-INDEXED!
    // ═══════════════════════════════════════════════════════════════════════════
    // CUDA Reference: KerPredicates.cu:166
    //   const int tetIdx = vertexTetArr[ idx ];  // idx is POSITION in uninserted array
    //
    // vert_tet[idx] where idx is the position (0, 1, 2, ..., num_uninserted-1)
    // NOT vert_tet[vert_idx] where vert_idx is the actual vertex ID!
    //
    // This indexing bug keeps reappearing during debugging. DO NOT CHANGE IT!
    // It causes subtle bugs where vertices read the wrong tetrahedron.
    // ═══════════════════════════════════════════════════════════════════════════
    let tet_idx = vert_tet[idx];  // POSITION-indexed! DO NOT use vert_tet[vert_idx]!

    if tet_idx == INVALID {
        return;
    }
    if (tet_info[tet_idx] & TET_ALIVE) == 0u {
        return;
    }

    // NOW get the vertex ID from uninserted array (after tet validation)
    let vert_idx = uninserted[idx];
        return;
    }
    if (tet_info[tet_idx] & TET_ALIVE) == 0u {
        return;
    }

    // Use insphere determinant (circumcenter) for voting
    // Port of CUDA kerVoteForPoint (KerPredicates.cu:160-200)
    let tet = tets[tet_idx];
    let p = points[vert_idx].xyz;  // vert_idx is vertex ID, used for geometry

    // Compute insphere determinant
    let pa = points[tet.x].xyz;
    let pb = points[tet.y].xyz;
    let pc = points[tet.z].xyz;
    let pd = points[tet.w].xyz;

    let result = insphere_fast(
        pa.x, pa.y, pa.z,
        pb.x, pb.y, pb.z,
        pc.x, pc.y, pc.z,
        pd.x, pd.y, pd.z,
        p.x, p.y, p.z
    );

    var sphere_val = result.x;  // det (positive = outside sphere)

    // CUDA line 186-187: Clamp negative values to 0
    if sphere_val < 0.0 {
        sphere_val = 0.0;
    }

    // CUDA line 189: Convert to int bits for atomic comparison
    let ival = bitcast<i32>(sphere_val);

    // CUDA line 191: Store sphere value for this vertex
    vert_sphere[idx] = ival;

    // CUDA line 196: Vote with atomicMax (furthest outside wins)
    atomicMax(&tet_vote[tet_idx], ival);
}

// Second pass: Pick winner using atomicMin
// Port of kerPickWinnerPoint from KerDivision.cu:311-335

@group(0) @binding(0) var<storage, read> vert_sphere2: array<i32>;
@group(0) @binding(1) var<storage, read> tet_vote2: array<i32>;
@group(0) @binding(2) var<storage, read> vert_tet2: array<u32>;
@group(0) @binding(3) var<storage, read_write> tet_to_vert: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read> uninserted2: array<u32>;
@group(0) @binding(5) var<uniform> params2: vec4<u32>; // x = num_uninserted

@compute @workgroup_size(64)
fn pick_winner_point(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let idx = gid.x;
    let num_uninserted = params2.x;

    if idx >= num_uninserted {
        return;
    }

    // CUDA line 323: Read vertex's sphere value (position-indexed)
    let vert_sval = vert_sphere2[idx];

    // ═══════════════════════════════════════════════════════════════════════════
    // ⚠️  CRITICAL: vert_tet2 is POSITION-INDEXED, NOT VERTEX-INDEXED!
    // ═══════════════════════════════════════════════════════════════════════════
    // CUDA Reference: KerDivision.cu:324
    //   const int tetIdx = vertexTetArr[ idx ];  // idx is POSITION in uninserted array
    //
    // vert_tet2[idx] where idx is the position (0, 1, 2, ..., num_uninserted-1)
    // NOT vert_tet2[vert_idx] where vert_idx is the actual vertex ID!
    //
    // This indexing bug keeps reappearing during debugging. DO NOT CHANGE IT!
    // ═══════════════════════════════════════════════════════════════════════════
    let tet_idx = vert_tet2[idx];  // POSITION-indexed! DO NOT use vert_tet2[vert_idx]!

    if tet_idx == INVALID {
        return;
    }

    // Get vertex ID from uninserted array (idx is position, not vertex ID)
    let vert_idx = uninserted2[idx];
        return;
    }

    // CUDA line 325: Read winning sphere value for that tet
    let win_sval = tet_vote2[tet_idx];

    // CUDA line 329-330: If this vertex matches winning value, compete with atomicMin
    if win_sval == vert_sval {
        atomicMin(&tet_to_vert[tet_idx], idx);
    }
}

// Third pass: Build insert list from winners
// Port of kerNegateInsertedVerts from KerDivision.cu:872-889

@group(0) @binding(0) var<storage, read> vert_tet3: array<u32>;
@group(0) @binding(1) var<storage, read> tet_to_vert3: array<u32>;
@group(0) @binding(2) var<storage, read> tet_info3: array<u32>;
@group(0) @binding(3) var<storage, read_write> insert_list: array<vec2<u32>>;
@group(0) @binding(4) var<storage, read_write> counters: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read> uninserted3: array<u32>;
@group(0) @binding(6) var<uniform> params3: vec4<u32>; // x = num_uninserted

const COUNTER_INSERTED: u32 = 2u;

@compute @workgroup_size(64)
fn build_insert_list(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let idx = gid.x;
    let num_uninserted = params3.x;

    if idx >= num_uninserted {
        return;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // ⚠️  CRITICAL: vert_tet3 is POSITION-INDEXED, NOT VERTEX-INDEXED!
    // ═══════════════════════════════════════════════════════════════════════════
    // CUDA Reference: KerDivision.cu:880
    //   const int tetIdx = vertTetVec._arr[ idx ];  // idx is POSITION
    //
    // This indexing bug keeps reappearing during debugging. DO NOT CHANGE IT!
    // ═══════════════════════════════════════════════════════════════════════════
    let tet_idx = vert_tet3[idx];  // POSITION-indexed! DO NOT use vert_tet3[vert_idx]!

    if tet_idx == INVALID {
        return;
    }
    if (tet_info3[tet_idx] & TET_ALIVE) == 0u {
        return;
    }

    // Get vertex ID from uninserted array (idx is position, not vertex ID)
    let vert_idx = uninserted3[idx];
        return;
    }

    if (tet_info3[tet_idx] & TET_ALIVE) == 0u {
        return;
    }

    // CUDA line 884: Check if this vertex is the winner for its tet
    if tet_to_vert3[tet_idx] == idx {
        // This vertex won - add to insert list (tet_idx, vertex_id)
        let slot = atomicAdd(&counters[COUNTER_INSERTED], 1u);
        insert_list[slot] = vec2<u32>(tet_idx, vert_idx);  // Store vertex ID, not position!
    }
}
