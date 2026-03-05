// split_points.wgsl
// Port of kerSplitPointsFast from gDel3D/GDelFlipping/src/gDel3D/GPU/KerPredicates.cu:285-303
//
// Updates vert_tet for uninserted vertices whose containing tets are being split.
// Uses an orient3d decision tree to determine which of the 4 new split tets contains each vertex.

@group(0) @binding(0) var<storage, read> points: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> tets: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> tet_opp: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> tet_info: array<u32>;
@group(0) @binding(4) var<storage, read_write> vert_tet: array<u32>;
@group(0) @binding(5) var<storage, read> free_arr: array<u32>;
@group(0) @binding(6) var<storage, read> vert_free_arr: array<u32>;
@group(0) @binding(7) var<storage, read_write> counters: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read> uninserted: array<u32>;
@group(0) @binding(9) var<storage, read> tet_to_vert: array<u32>; // Maps tet idx -> position in uninserted array

const MEAN_VERTEX_DEGREE: u32 = 8u;
const INVALID: u32 = 0xFFFFFFFFu;

// SplitFaces[11][3] - Face vertices for orient3d decision tree
// From GPUDecl.h:167-172
const SPLIT_FACES: array<array<u32, 3>, 11> = array(
    array(0u, 1u, 4u),  // 0
    array(0u, 3u, 4u),  // 1
    array(0u, 2u, 4u),  // 2
    array(2u, 3u, 4u),  // 3
    array(1u, 3u, 4u),  // 4
    array(1u, 2u, 4u),  // 5
    array(2u, 3u, 4u),  // 6
    array(1u, 3u, 2u),  // 7
    array(0u, 2u, 3u),  // 8
    array(0u, 3u, 1u),  // 9
    array(0u, 1u, 2u),  // 10
);

// SplitNext[11][2] - Next face index based on orient3d result
// From GPUDecl.h:174-179
// [face][0] = next if orient positive, [face][1] = next if orient negative/zero
const SPLIT_NEXT: array<array<u32, 2>, 11> = array(
    array(1u, 2u),    // 0
    array(3u, 4u),    // 1
    array(5u, 6u),    // 2
    array(7u, 8u),    // 3
    array(9u, 7u),    // 4
    array(7u, 10u),   // 5
    array(7u, 8u),    // 6
    array(1u, 0u),    // 7
    array(2u, 0u),    // 8
    array(3u, 0u),    // 9
    array(4u, 0u),    // 10
);

// --- Double-Double Arithmetic for Exact Predicates ---

struct DD {
    hi: f32,
    lo: f32,
}

fn two_sum(a: f32, b: f32) -> DD {
    let s = a + b;
    let v = s - a;
    let e = (a - (s - v)) + (b - v);
    return DD(s, e);
}

fn fast_two_sum(a: f32, b: f32) -> DD {
    let s = a + b;
    let e = b - (s - a);
    return DD(s, e);
}

fn two_product(a: f32, b: f32) -> DD {
    let p = a * b;
    let e = fma(a, b, -p);
    return DD(p, e);
}

fn dd_add(a: DD, b: DD) -> DD {
    let s = two_sum(a.hi, b.hi);
    let t = two_sum(a.lo, b.lo);
    var c = fast_two_sum(s.hi, s.lo + t.hi);
    c = fast_two_sum(c.hi, c.lo + t.lo);
    return c;
}

fn dd_sub(a: DD, b: DD) -> DD {
    return dd_add(a, DD(-b.hi, -b.lo));
}

fn dd_mul(a: DD, b: DD) -> DD {
    let p = two_product(a.hi, b.hi);
    let e = a.hi * b.lo + a.lo * b.hi + p.lo;
    return fast_two_sum(p.hi, e);
}

fn dd_from_f32(x: f32) -> DD {
    return DD(x, 0.0);
}

fn dd_sign(a: DD) -> i32 {
    if a.hi > 0.0 { return 1; }
    if a.hi < 0.0 { return -1; }
    if a.lo > 0.0 { return 1; }
    if a.lo < 0.0 { return -1; }
    return 0;
}

// Orient3d with double-double arithmetic
fn orient3d_dd(
    ax: f32, ay: f32, az: f32,
    bx: f32, by: f32, bz: f32,
    cx: f32, cy: f32, cz: f32,
    dx: f32, dy: f32, dz: f32,
) -> i32 {
    let adx = dd_from_f32(ax - dx);
    let ady = dd_from_f32(ay - dy);
    let adz = dd_from_f32(az - dz);
    let bdx = dd_from_f32(bx - dx);
    let bdy = dd_from_f32(by - dy);
    let bdz = dd_from_f32(bz - dz);
    let cdx = dd_from_f32(cx - dx);
    let cdy = dd_from_f32(cy - dy);
    let cdz = dd_from_f32(cz - dz);

    let t1 = dd_sub(dd_mul(bdy, cdz), dd_mul(bdz, cdy));
    let t2 = dd_sub(dd_mul(cdy, adz), dd_mul(cdz, ady));
    let t3 = dd_sub(dd_mul(ady, bdz), dd_mul(adz, bdy));

    let det = dd_add(dd_add(dd_mul(adx, t1), dd_mul(bdx, t2)), dd_mul(cdx, t3));
    return dd_sign(det);
}

// Fast orient3d with error bounds
fn orient3d_fast_check(
    ax: f32, ay: f32, az: f32,
    bx: f32, by: f32, bz: f32,
    cx: f32, cy: f32, cz: f32,
    dx: f32, dy: f32, dz: f32,
) -> vec2<f32> {
    let adx = ax - dx; let ady = ay - dy; let adz = az - dz;
    let bdx = bx - dx; let bdy = by - dy; let bdz = bz - dz;
    let cdx = cx - dx; let cdy = cy - dy; let cdz = cz - dz;

    let det = adx * (bdy * cdz - bdz * cdy)
            + bdx * (cdy * adz - cdz * ady)
            + cdx * (ady * bdz - adz * bdy);

    let eps = 5.96e-8;
    let permanent = abs(adx) * (abs(bdy * cdz) + abs(bdz * cdy))
                  + abs(bdx) * (abs(cdy * adz) + abs(cdz * ady))
                  + abs(cdx) * (abs(ady * bdz) + abs(adz * bdy));
    let err_bound = 7.77e-7 * permanent;

    if abs(det) > err_bound {
        return vec2<f32>(det, 0.0);
    }
    return vec2<f32>(det, 1.0);
}

// Exact orient3d with adaptive filter
fn orient3d_exact(
    ax: f32, ay: f32, az: f32,
    bx: f32, by: f32, bz: f32,
    cx: f32, cy: f32, cz: f32,
    dx: f32, dy: f32, dz: f32,
) -> i32 {
    let fast = orient3d_fast_check(ax, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz);
    if fast.y == 0.0 {
        if fast.x > 0.0 { return 1; }
        if fast.x < 0.0 { return -1; }
        return 0;
    }
    return orient3d_dd(ax, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz);
}

// SoS tie-breaking using vertex indices
fn sos_orient3d_index(va: u32, vb: u32, vc: u32, vd: u32) -> i32 {
    var indices = array<u32, 4>(va, vb, vc, vd);
    var parity = 0u;

    for (var i = 0u; i < 3u; i++) {
        for (var j = 0u; j < 3u - i; j++) {
            if indices[j] > indices[j + 1u] {
                let tmp = indices[j];
                indices[j] = indices[j + 1u];
                indices[j + 1u] = tmp;
                parity ^= 1u;
            }
        }
    }

    return select(-1, 1, parity == 0u);
}

// Orient3d with SoS
fn orient3d_with_sos(
    a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, d: vec3<f32>,
    va: u32, vb: u32, vc: u32, vd: u32
) -> i32 {
    let result = orient3d_exact(a.x, a.y, a.z, b.x, b.y, b.z, c.x, c.y, c.z, d.x, d.y, d.z);

    if result == 0 {
        return sos_orient3d_index(va, vb, vc, vd);
    }

    return result;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let vert_idx = global_id.x;
    let num_uninserted = arrayLength(&uninserted);

    if (vert_idx >= num_uninserted) {
        return;
    }

    // Track how many threads enter
    atomicAdd(&counters[5], 1u); // scratch[1]

    // Get current tet for this uninserted vertex
    var tet_idx = vert_tet[vert_idx];

    if (tet_idx == INVALID) {
        return; // Vertex not yet located
    }

    // Track how many have valid tet_idx
    atomicAdd(&counters[4], 1u); // scratch[2]

    // Check if this tet is being split
    let split_vert_position = tet_to_vert[tet_idx];

    if (split_vert_position == INVALID) {
        return; // Tet not being split, nothing to update
    }

    // Get vertex ID of the vertex being inserted (that's splitting this tet)
    let split_vertex = uninserted[split_vert_position];

    // Get the actual vertex ID for this uninserted point
    let vertex = uninserted[vert_idx];

    // Self-insertion check (CUDA: KerPredicates.cu:228)
    // Don't update vert_tet for the vertex that's being inserted
    if split_vertex == vertex {
        return;
    }

    // Load the original tet being split
    let tet = tets[tet_idx];

    // Build array of 5 vertices: 4 from original tet + 1 being inserted
    var tet_verts = array<u32, 5>(tet.x, tet.y, tet.z, tet.w, split_vertex);

    // Load points for all 5 vertices
    var pts = array<vec3<f32>, 5>(
        points[tet_verts[0]].xyz,
        points[tet_verts[1]].xyz,
        points[tet_verts[2]].xyz,
        points[tet_verts[3]].xyz,
        points[tet_verts[4]].xyz,
    );

    let pt_vertex = points[vertex].xyz;

    // Decision tree: 3 iterations to determine which of 4 new tets contains this vertex
    var face = 0u;

    for (var i = 0u; i < 3u; i++) {
        // Get face vertices for current face
        let fv = SPLIT_FACES[face];

        // Orient3d test: is vertex above or below this face?
        // Use exact predicates with SoS for degenerate cases
        let orient = orient3d_with_sos(
            pts[fv[0]],
            pts[fv[1]],
            pts[fv[2]],
            pt_vertex,
            tet_verts[fv[0]],
            tet_verts[fv[1]],
            tet_verts[fv[2]],
            vertex
        );

        // Navigate to next face based on orientation
        if (orient > 0) {
            face = SPLIT_NEXT[face][0]; // Positive orientation
        } else {
            face = SPLIT_NEXT[face][1]; // Negative or zero orientation
        }
    }

    // After 3 iterations, face should be in range 7-10
    // Map to actual tet index via free_arr
    // The split operation allocates 4 tets from the split vertex's free list
    // They are at positions: free_idx, free_idx-1, free_idx-2, free_idx-3
    // where free_idx = (split_vertex + 1) * MEAN_VERTEX_DEGREE - 1 (top of stack)

    if (face >= 7u && face <= 10u) {
        let free_idx = (split_vertex + 1u) * MEAN_VERTEX_DEGREE - 1u;
        let offset = face - 7u; // 0, 1, 2, or 3

        // Bounds check
        if (free_idx >= offset && free_idx - offset < arrayLength(&free_arr)) {
            let new_tet = free_arr[free_idx - offset];
            vert_tet[vert_idx] = new_tet;
            // Increment counter to track updates
            atomicAdd(&counters[6], 1u); // Use scratch[0] counter
        }
    }
}
