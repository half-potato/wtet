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
@group(0) @binding(7) var<uniform> params: vec4<u32>; // x = num_uninserted, y = inf_idx (infinity vertex)

const INVALID: u32 = 0xFFFFFFFFu;
const NO_VOTE: i32 = i32(0x80000000u);  // Minimum i32 for atomicMax voting
const TET_ALIVE: u32 = 1u;

// TetViAsSeenFrom[vi] returns the 3 other vertices in order when viewed from vertex vi
// CUDA Reference: CommonTypes.h:128-133
// NOTE: Cannot use as constant array (variable indexing causes SIGSEGV)
// Implemented as explicit branches in infinity handling code below

// Fast orient3d: returns determinant (positive = point is on left side of plane)
// CUDA Reference: KerShewchuk.h orient3dfast()
fn orient3d_fast(
    ax: f32, ay: f32, az: f32,
    bx: f32, by: f32, bz: f32,
    cx: f32, cy: f32, cz: f32,
    dx: f32, dy: f32, dz: f32,
) -> f32 {
    let adx = ax - dx;
    let bdx = bx - dx;
    let cdx = cx - dx;
    let ady = ay - dy;
    let bdy = by - dy;
    let cdy = cy - dy;
    let adz = az - dz;
    let bdz = bz - dz;
    let cdz = cz - dz;

    return (adx * (bdy * cdz - bdz * cdy))
         + (bdx * (cdy * adz - cdz * ady))
         + (cdx * (ady * bdz - adz * bdy));
}

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

@compute @workgroup_size(256)
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
    // Bounds check: defensive programming (not in CUDA but prevents out-of-bounds)
    if tet_idx >= arrayLength(&tets) {
        return;
    }
    if (tet_info[tet_idx] & TET_ALIVE) == 0u {
        return;
    }

    // NOW get the vertex ID from uninserted array (after tet validation)
    let vert_idx = uninserted[idx];

    // Use insphere determinant (circumcenter) for voting
    // Port of CUDA kerVoteForPoint (KerPredicates.cu:160-200)
    // With infinity handling from KerPredWrapper.h:888-895
    let tet = tets[tet_idx];
    let p = points[vert_idx].xyz;  // vert_idx is vertex ID, used for geometry
    let inf_idx = params.y;  // Infinity vertex index

    // Validate inf_idx: defensive programming (not in CUDA but prevents corruption)
    if inf_idx >= arrayLength(&points) {
        return;
    }

    var sphere_val: f32;

    // Check if tet contains infinity vertex
    // CUDA Reference: KerPredWrapper.h:889-895 (inSphereDet)
    if (tet.x == inf_idx || tet.y == inf_idx || tet.z == inf_idx || tet.w == inf_idx) {
        // Tet contains infinity - use orient3d instead of insphere
        // CRITICAL: Cannot use variable indexing into arrays (causes SIGSEGV)
        // Must use explicit branches for TetViAsSeenFrom lookup

        var v0: u32;
        var v1: u32;
        var v2: u32;

        // TetViAsSeenFrom[inf_vi] gives the 3 other vertices in order
        if (tet.x == inf_idx) {
            // From vertex 0: {1, 3, 2}
            v0 = tet.y;
            v1 = tet.w;
            v2 = tet.z;
        } else if (tet.y == inf_idx) {
            // From vertex 1: {0, 2, 3}
            v0 = tet.x;
            v1 = tet.z;
            v2 = tet.w;
        } else if (tet.z == inf_idx) {
            // From vertex 2: {0, 3, 1}
            v0 = tet.x;
            v1 = tet.w;
            v2 = tet.y;
        } else {
            // From vertex 3: {0, 1, 2}
            v0 = tet.x;
            v1 = tet.y;
            v2 = tet.z;
        }

        let pa = points[v0].xyz;
        let pb = points[v1].xyz;
        let pc = points[v2].xyz;

        // Use orient3d to check convexity (negated, CUDA line 895)
        let det = orient3d_fast(
            pa.x, pa.y, pa.z,
            pb.x, pb.y, pb.z,
            pc.x, pc.y, pc.z,
            p.x, p.y, p.z
        );
        sphere_val = -det;  // Negate to match CUDA
    } else {
        // Regular tet - use insphere
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

        // CRITICAL: Negate determinant to match CUDA convention!
        // CUDA (KerPredWrapper.h:898): det = -insphereDet(...)
        // After negation: positive = inside sphere (should insert), negative = outside
        sphere_val = -result.x;
    }

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
@group(0) @binding(5) var<storage, read> tet_info2: array<u32>;
@group(0) @binding(6) var<uniform> params2: vec4<u32>; // x = num_uninserted

@compute @workgroup_size(256)
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
    // Bounds check: defensive programming (not in CUDA but prevents out-of-bounds)
    if tet_idx >= arrayLength(&tet_vote2) {
        return;
    }
    // Alive check: for consistency with vote_for_point and build_insert_list
    // (Technically redundant since vote_for_point already filtered, but defensive)
    if (tet_info2[tet_idx] & TET_ALIVE) == 0u {
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

@compute @workgroup_size(256)
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
    // Bounds check: defensive programming (not in CUDA but prevents out-of-bounds)
    if tet_idx >= arrayLength(&tet_info3) {
        return;
    }
    if (tet_info3[tet_idx] & TET_ALIVE) == 0u {
        return;
    }

    // Get vertex ID from uninserted array (idx is position, not vertex ID)
    let vert_idx = uninserted3[idx];

    // ┌────────────────────────────────────────────────────────────────────────┐
    // │ CRITICAL: insert_list stores (tet_idx, POSITION), NOT (tet_idx, vert_idx)! │
    // │ Position = idx in uninserted array, used by split.wgsl and compaction │
    // │ CUDA ref: kerPickWinnerPoint (KerDivision.cu:311-335) uses vertex INDEX │
    // │ DO NOT CHANGE TO vert_idx - breaks split and compaction!              │
    // └────────────────────────────────────────────────────────────────────────┘
    // Check if this vertex is the winner for its tet
    if tet_to_vert3[tet_idx] == idx {
        // This vertex won - add to insert list with POSITION (idx), not vertex ID
        let slot = atomicAdd(&counters[COUNTER_INSERTED], 1u);
        insert_list[slot] = vec2<u32>(tet_idx, idx);  // Store position (idx), not vert_idx!
    }
}
