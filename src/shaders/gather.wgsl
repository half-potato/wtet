// Kernel: Gather vertices that failed the Delaunay property.
//
// After all GPU insertion+flipping iterations, check every tet for insphere
// violations. Collect the offending vertices into a compact buffer for
// CPU Phase 2 star splaying.

@group(0) @binding(0) var<storage, read> points: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> tets: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> tet_opp: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> tet_info: array<u32>;
@group(0) @binding(4) var<storage, read_write> failed_verts: array<u32>;
@group(0) @binding(5) var<storage, read_write> counters: array<atomic<u32>>;
@group(0) @binding(6) var<uniform> params: vec4<u32>; // x = max_tets, y = num_points

const INVALID: u32 = 0xFFFFFFFFu;
const TET_ALIVE: u32 = 1u;
const COUNTER_FAILED: u32 = 3u;

fn insphere_simple(
    a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, d: vec3<f32>, e: vec3<f32>
) -> f32 {
    let ae = a - e; let be = b - e; let ce = c - e; let de = d - e;

    let ab = ae.x * be.y - be.x * ae.y;
    let bc = be.x * ce.y - ce.x * be.y;
    let cd = ce.x * de.y - de.x * ce.y;
    let da = de.x * ae.y - ae.x * de.y;
    let ac = ae.x * ce.y - ce.x * ae.y;
    let bd = be.x * de.y - de.x * be.y;

    let abc = ae.z * bc - be.z * ac + ce.z * ab;
    let bcd = be.z * cd - ce.z * bd + de.z * bc;
    let cda = ce.z * da + de.z * ac + ae.z * cd;
    let dab = de.z * ab + ae.z * bd + be.z * da;

    let al = dot(ae, ae); let bl = dot(be, be);
    let cl = dot(ce, ce); let dl = dot(de, de);

    return (dl * abc - cl * dab) + (bl * cda - al * bcd);
}

fn orient3d_simple(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, d: vec3<f32>) -> f32 {
    let ad = a - d; let bd = b - d; let cd = c - d;
    return ad.x * (bd.y * cd.z - bd.z * cd.y)
         + bd.x * (cd.y * ad.z - cd.z * ad.y)
         + cd.x * (ad.y * bd.z - ad.z * bd.y);
}

fn tet_vertex(tet: vec4<u32>, i: u32) -> u32 {
    switch i {
        case 0u: { return tet.x; }
        case 1u: { return tet.y; }
        case 2u: { return tet.z; }
        default: { return tet.w; }
    }
}

fn opp_entry(opp: vec4<u32>, i: u32) -> u32 {
    switch i {
        case 0u: { return opp.x; }
        case 1u: { return opp.y; }
        case 2u: { return opp.z; }
        default: { return opp.w; }
    }
}

@compute @workgroup_size(64)
fn gather_failed(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let tet_idx = gid.x;
    let max_tets = params.x;
    let num_points = params.y;

    if tet_idx >= max_tets {
        return;
    }

    if (tet_info[tet_idx] & TET_ALIVE) == 0u {
        return;
    }

    let tet = tets[tet_idx];
    let opp = tet_opp[tet_idx];

    // For each face, check if the opposite vertex violates Delaunay
    for (var f = 0u; f < 4u; f++) {
        let opp_packed = opp_entry(opp, f);
        if opp_packed == INVALID {
            continue;
        }

        let opp_tet_idx = opp_packed >> 5u;
        let opp_face = opp_packed & 3u;

        // Only check if our index < neighbour index (avoid double-counting)
        if tet_idx >= opp_tet_idx {
            continue;
        }

        let opp_tet = tets[opp_tet_idx];
        let opposite_vert = tet_vertex(opp_tet, opp_face);

        // Skip super-tet vertices
        if opposite_vert >= num_points {
            continue;
        }

        // Also skip if any tet vertex is a super-tet vertex
        var has_super = false;
        for (var i = 0u; i < 4u; i++) {
            if tet_vertex(tet, i) >= num_points {
                has_super = true;
            }
        }
        if has_super {
            continue;
        }

        let v0 = points[tet.x].xyz;
        let v1 = points[tet.y].xyz;
        let v2 = points[tet.z].xyz;
        let v3 = points[tet.w].xyz;
        let pe = points[opposite_vert].xyz;

        let o = orient3d_simple(v0, v1, v2, v3);

        var is_inside: f32;
        if o > 0.0 {
            is_inside = insphere_simple(v0, v1, v2, v3, pe);
        } else {
            is_inside = -insphere_simple(v0, v2, v1, v3, pe);
        }

        if is_inside > 0.0 {
            // Delaunay violation! Mark the opposite vertex as failed.
            let slot = atomicAdd(&counters[COUNTER_FAILED], 1u);
            failed_verts[slot] = opposite_vert;
        }
    }
}
