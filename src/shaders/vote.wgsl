// Port of kerVoteForPoint from gDel3D/GPU/KerPredicates.cu (lines 151-200)
// Each uninserted point votes for its containing tet based on insphere/distance

@group(0) @binding(0) var<storage, read> points: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> uninserted: array<u32>; // vertexArr._arr in original
@group(0) @binding(2) var<storage, read> vert_tet: array<u32>; // vertexTetArr in original - indexed by idx!
@group(0) @binding(3) var<storage, read> tets: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read_write> vert_sphere: array<atomic<i32>>; // vertSphereArr - indexed by idx!
@group(0) @binding(5) var<storage, read_write> tet_sphere: array<atomic<i32>>; // tetSphereArr in original
@group(0) @binding(6) var<uniform> params: vec4<u32>; // x = num_uninserted, y = insertion_rule

const INSERTION_CIRCUMCENTER: u32 = 0u;
const INSERTION_CENTROID: u32 = 1u;
const INSERTION_RANDOM: u32 = 2u;

// Compute distance from point to tet centroid (simplified version)
fn dist_to_centroid(tet: vec4<u32>, vert: u32) -> f32 {
    let p = points[vert].xyz;
    let v0 = points[tet.x].xyz;
    let v1 = points[tet.y].xyz;
    let v2 = points[tet.z].xyz;
    let v3 = points[tet.w].xyz;

    let centroid = (v0 + v1 + v2 + v3) * 0.25;
    let diff = p - centroid;
    return length(diff);
}

// Compute insphere determinant (simplified - exact arithmetic needed for production)
fn in_sphere_det(tet: vec4<u32>, vert: u32) -> f32 {
    // TODO: This should use exact arithmetic like the original
    // For now, use a simple distance metric
    return dist_to_centroid(tet, vert);
}

// Simple hash function for random insertion
fn hash(v: u32) -> f32 {
    var x = v;
    x = ((x >> 16u) ^ x) * 0x45d9f3bu;
    x = ((x >> 16u) ^ x) * 0x45d9f3bu;
    x = (x >> 16u) ^ x;
    return f32(x) / 4294967295.0;
}

// Direct port of kerVoteForPoint (lines 151-200)
@compute @workgroup_size(64)
fn vote_for_point(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let idx = gid.x;
    let num_uninserted = params.x;
    let ins_rule = params.y;

    // Line 162: for ( int idx = getCurThreadIdx(); idx < vertexArr._num; idx += getThreadNum() )
    if idx >= num_uninserted {
        return;
    }

    // Lines 164-168: EXACT MATCH to original
    let tet_idx = vert_tet[idx];              // Line 166: vertexTetArr[idx]
    let tet = tets[tet_idx];                  // Line 167: tetArr[tetIdx]
    let vert = uninserted[idx];               // Line 168: vertexArr._arr[idx]
    var sval: f32;

    // Lines 171-182: switch on insertion rule
    switch ins_rule {
        case INSERTION_CIRCUMCENTER: {
            sval = in_sphere_det(tet, vert);
        }
        case INSERTION_CENTROID: {
            sval = dist_to_centroid(tet, vert);
        }
        case INSERTION_RANDOM: {
            sval = hash(vert);
        }
        default: {
            sval = in_sphere_det(tet, vert);
        }
    }

    // Lines 186-187: Sanitize
    if sval < 0.0 {
        sval = 0.0;
    }

    // Line 189: Convert to int (bitwise)
    let ival = bitcast<i32>(sval);

    // Line 191: Store sphere value - INDEXED BY idx, not vert!
    atomicStore(&vert_sphere[idx], ival);

    // Lines 195-196: Vote (with optimization to reduce atomic cost)
    if atomicLoad(&tet_sphere[tet_idx]) < ival {
        atomicMax(&tet_sphere[tet_idx], ival);
    }
}
