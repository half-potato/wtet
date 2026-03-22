// Kernel: Create the initial 5-tet structure (1 base + 4 boundary tets).
//
// Port of CUDA's kerMakeFirstTetra (KerDivision.cu:46-95)
//
// The base tetrahedron uses 4 super-tet vertices placed far enough to contain
// the entire [0,1]³ bounding box. The 4 boundary tets connect to an infinity
// vertex (index N+4) to form a closed boundary.
//
// Dispatch: (1, 1, 1) — single invocation.

@group(0) @binding(0) var<storage, read> points: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> tets: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> tet_opp: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> tet_info: array<u32>;
@group(0) @binding(4) var<storage, read_write> vert_tet: array<u32>;
@group(0) @binding(5) var<storage, read_write> free_arr: array<u32>;
@group(0) @binding(6) var<storage, read_write> vert_free_arr: array<u32>;
@group(0) @binding(7) var<storage, read_write> counters: array<atomic<u32>>;
@group(0) @binding(8) var<uniform> params: vec4<u32>; // x = num_points, y = max_tets

const INVALID: u32 = 0xFFFFFFFFu;
const TET_ALIVE: u32 = 1u;

// 5-bit TetOpp encoding (CUDA CommonTypes.h:248-265)
fn encode_opp(tet_idx: u32, face: u32) -> u32 {
    return (tet_idx << 5u) | (face & 3u);
}

// CUDA adjacency tables from KerDivision.cu:46-95
// WARNING: WGSL doesn't support variable array indexing - use explicit if/else
fn get_opp_tet(tet_i: u32, face_j: u32) -> u32 {
    if tet_i == 0u {
        if face_j == 0u { return 1u; }
        else if face_j == 1u { return 2u; }
        else if face_j == 2u { return 3u; }
        else { return 4u; }
    } else if tet_i == 1u {
        if face_j == 0u { return 2u; }
        else if face_j == 1u { return 3u; }
        else if face_j == 2u { return 4u; }
        else { return 0u; }
    } else if tet_i == 2u {
        if face_j == 0u { return 1u; }
        else if face_j == 1u { return 4u; }
        else if face_j == 2u { return 3u; }
        else { return 0u; }
    } else if tet_i == 3u {
        if face_j == 0u { return 1u; }
        else if face_j == 1u { return 2u; }
        else if face_j == 2u { return 4u; }
        else { return 0u; }
    } else { // tet_i == 4u
        if face_j == 0u { return 1u; }
        else if face_j == 1u { return 3u; }
        else if face_j == 2u { return 2u; }
        else { return 0u; }
    }
}

fn get_opp_vi(tet_i: u32, face_j: u32) -> u32 {
    if tet_i == 0u {
        // All faces of tet 0 connect to face 3 of neighbors
        return 3u;
    } else if tet_i == 1u {
        if face_j == 0u { return 0u; }
        else if face_j == 1u { return 0u; }
        else if face_j == 2u { return 0u; }
        else { return 0u; }
    } else if tet_i == 2u {
        if face_j == 0u { return 0u; }
        else if face_j == 1u { return 2u; }
        else if face_j == 2u { return 1u; }
        else { return 1u; }
    } else if tet_i == 3u {
        if face_j == 0u { return 1u; }
        else if face_j == 1u { return 2u; }
        else if face_j == 2u { return 1u; }
        else { return 2u; }
    } else { // tet_i == 4u
        if face_j == 0u { return 2u; }
        else if face_j == 1u { return 2u; }
        else if face_j == 2u { return 1u; }
        else { return 3u; }
    }
}

@compute @workgroup_size(1)
fn make_first_tetra() {
    let n = params.x;

    // Virtual vertices: n, n+1, n+2, n+3 (super-tet), n+4 (infinity)
    var sv0 = n;
    var sv1 = n + 1u;
    let v2 = n + 2u;
    let v3 = n + 3u;
    let inf_idx = n + 4u;

    // Orient3d check — swap v0,v1 if negative to ensure positive orientation.
    // CUDA: GpuDelaunay.cu:284-293
    //   RealType ori = orient3dzero(p0, p1, p2, p3);
    //   if (ortToOrient(ori) == OrientNeg) std::swap(v0, v1);
    let p0 = points[sv0].xyz;
    let p1 = points[sv1].xyz;
    let p2 = points[v2].xyz;
    let p3 = points[v3].xyz;
    let ad = p0 - p3; let bd = p1 - p3; let cd = p2 - p3;
    let orient = ad.x * (bd.y * cd.z - bd.z * cd.y)
               + bd.x * (cd.y * ad.z - cd.z * ad.y)
               + cd.x * (ad.y * bd.z - ad.z * bd.y);
    // CUDA (GpuDelaunay.cu:292): swaps when ortToOrient(ori) == OrientNeg
    // ortToOrient negates Shewchuk sign: det > 0 → OrientNeg
    // So swap when Shewchuk orient3d > 0, ensuring orient3d < 0 after swap.
    // For orient3d < 0: insphere < 0 = inside circumsphere = violation.
    if orient > 0.0 {
        let tmp = sv0;
        sv0 = sv1;
        sv1 = tmp;
    }

    // Create 5 tets matching CUDA's kerMakeFirstTetra (KerDivision.cu:46-95)
    // Uses (potentially swapped) sv0, sv1 to ensure positive orient3d.
    tets[0] = vec4<u32>(sv0, sv1, v2, v3);         // Base tet
    tets[1] = vec4<u32>(sv1, v2, v3, inf_idx);     // Boundary face 0
    tets[2] = vec4<u32>(sv0, v3, v2, inf_idx);     // Boundary face 1
    tets[3] = vec4<u32>(sv0, sv1, v3, inf_idx);    // Boundary face 2
    tets[4] = vec4<u32>(sv0, v2, sv1, inf_idx);    // Boundary face 3

    // Set adjacency for all 5 tets
    for (var t = 0u; t < 5u; t++) {
        for (var f = 0u; f < 4u; f++) {
            let nei_tet = get_opp_tet(t, f);
            let nei_face = get_opp_vi(t, f);
            atomicStore(&tet_opp[t * 4u + f], encode_opp(nei_tet, nei_face));
        }
    }

    // Mark all 5 tets as alive
    for (var i = 0u; i < 5u; i++) {
        tet_info[i] = TET_ALIVE;
    }

    // Set counters (free stack was pre-filled by CPU with max_tets-1 entries)
    let max_tets = params.y;
    atomicStore(&counters[0], max_tets - 5u); // free_count = max_tets - 5 (5 tets now used)
    atomicStore(&counters[1], 5u);            // active_count = 5
    atomicStore(&counters[2], 0u);            // inserted_count = 0
    atomicStore(&counters[3], 0u);            // failed_count = 0
}

// OBSOLETE: init_vert_tet shader pass (no longer dispatched after optimization)
// vert_tet is now initialized on CPU in buffers.rs for ~10ms speedup.
// This entry point must remain for pipeline creation, but dispatch_init no longer calls it.
@compute @workgroup_size(256)
fn init_vert_tet(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let n = params.x;
    let idx = gid.x;

    if idx >= n {
        return;
    }

    // All input points start in tet 0 (base tet)
    vert_tet[idx] = 0u;
}
