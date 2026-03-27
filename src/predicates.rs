//! Exact geometric predicates using Shewchuk adaptive expansion arithmetic.
//!
//! These are used as test oracles to verify the GPU double-double predicates.
//! `orient3d` and `insphere` delegate to the `geometry-predicates` crate which
//! provides exact adaptive implementations. The lifted variants (for the SoS
//! cascade) use expansion arithmetic primitives from the same crate, ported from
//! gDel3D/CPU/predicates.cpp.

use geometry_predicates::predicates::{
    fast_expansion_sum_zeroelim, scale_expansion_zeroelim, two_product, two_two_diff,
};

/// Exact orient3d via Shewchuk adaptive arithmetic (geometry-predicates crate).
pub fn orient3d(a: [f64; 3], b: [f64; 3], c: [f64; 3], d: [f64; 3]) -> f64 {
    geometry_predicates::orient3d(a, b, c, d)
}

/// Exact insphere via Shewchuk adaptive arithmetic (geometry-predicates crate).
pub fn insphere(a: [f64; 3], b: [f64; 3], c: [f64; 3], d: [f64; 3], e: [f64; 3]) -> f64 {
    geometry_predicates::insphere(a, b, c, d, e)
}

/// Compute circumcenter of tetrahedron (a, b, c, d).
/// Returns None if the tet is degenerate (zero volume).
pub fn circumcenter(a: [f64; 3], b: [f64; 3], c: [f64; 3], d: [f64; 3]) -> Option<[f64; 3]> {
    let ax = a[0] - d[0];
    let ay = a[1] - d[1];
    let az = a[2] - d[2];
    let bx = b[0] - d[0];
    let by = b[1] - d[1];
    let bz = b[2] - d[2];
    let cx = c[0] - d[0];
    let cy = c[1] - d[1];
    let cz = c[2] - d[2];

    let det = ax * (by * cz - bz * cy)
        + bx * (cy * az - cz * ay)
        + cx * (ay * bz - az * by);

    if det.abs() < 1e-30 {
        return None;
    }

    let a2 = ax * ax + ay * ay + az * az;
    let b2 = bx * bx + by * by + bz * bz;
    let c2 = cx * cx + cy * cy + cz * cz;

    let inv_det = 0.5 / det;

    let ux = (a2 * (by * cz - bz * cy) + b2 * (cy * az - cz * ay) + c2 * (ay * bz - az * by))
        * inv_det;
    let uy = (a2 * (bz * cx - bx * cz) + b2 * (cz * ax - cx * az) + c2 * (az * bx - ax * bz))
        * inv_det;
    let uz = (a2 * (bx * cy - by * cx) + b2 * (cx * ay - cy * ax) + c2 * (ax * by - ay * bx))
        * inv_det;

    Some([ux + d[0], uy + d[1], uz + d[2]])
}

/// Sign of a value: -1, 0, or 1.
pub fn sign(x: f64) -> i32 {
    if x > 0.0 {
        1
    } else if x < 0.0 {
        -1
    } else {
        0
    }
}

/// 2D orientation test. Returns positive if c is to the left of line ab,
/// negative if to the right, zero if collinear.
fn orient2d(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
    (a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) * (b[0] - c[0])
}

/// CUDA ortToOrient: det < 0 → OrientPos (+1), det > 0 → OrientNeg (-1)
fn ort_to_orient(det: f64) -> i32 {
    if det < 0.0 {
        1
    } else if det > 0.0 {
        -1
    } else {
        0
    }
}

/// CUDA sphToOrient: det > 0 → OrientPos (+1), det < 0 → OrientNeg (-1)
fn sph_to_orient(det: f64) -> i32 {
    if det > 0.0 {
        1
    } else if det < 0.0 {
        -1
    } else {
        0
    }
}

/// Orient3D with SoS tie-breaking.
///
/// Ported from CUDA PredWrapper.cpp doOrient3DSoS / doOrient3DSoSOnly (lines 104-261).
/// Returns ortToOrient(det): +1 if raw det < 0, -1 if raw det > 0.
/// Never returns 0 (SoS guarantees non-degeneracy).
///
/// No infinity handling — CPU code doesn't use infinity vertices.
pub fn orient3d_sos(
    a: [f64; 3], b: [f64; 3], c: [f64; 3], d: [f64; 3],
    a_idx: u32, b_idx: u32, c_idx: u32, d_idx: u32,
) -> i32 {
    let det = orient3d(a, b, c, d);
    // orient3d is now exact (Shewchuk adaptive) — non-zero means non-degenerate.
    if det != 0.0 {
        return ort_to_orient(det);
    }

    // SoS: sort 4 vertices by index, track swap parity
    let mut v = [a_idx, b_idx, c_idx, d_idx];
    let mut p = [a, b, c, d];
    let mut pn: f64 = 1.0;

    // Selection sort (matching CUDA)
    for i in 0..3 {
        let mut min_i = i;
        for j in (i + 1)..4 {
            if v[j] < v[min_i] {
                min_i = j;
            }
        }
        if min_i != i {
            v.swap(i, min_i);
            p.swap(i, min_i);
            pn = -pn;
        }
    }

    // 14-depth cascade (CUDA PredWrapper.cpp:150-247)
    let mut result: f64 = 0.0;
    let mut depth = 0;

    while depth < 14 {
        match depth {
            0 => {
                // orient2d on xy of p[1],p[2],p[3]
                result = orient2d([p[1][0], p[1][1]], [p[2][0], p[2][1]], [p[3][0], p[3][1]]);
            }
            1 => {
                // orient2d on xz of p[1],p[2],p[3]
                result = orient2d([p[1][0], p[1][2]], [p[2][0], p[2][2]], [p[3][0], p[3][2]]);
            }
            2 => {
                // orient2d on yz of p[1],p[2],p[3]
                result = orient2d([p[1][1], p[1][2]], [p[2][1], p[2][2]], [p[3][1], p[3][2]]);
            }
            3 => {
                // orient2d on xy of p[0],p[2],p[3]
                result = orient2d([p[0][0], p[0][1]], [p[2][0], p[2][1]], [p[3][0], p[3][1]]);
            }
            4 => {
                result = p[2][0] - p[3][0];
            }
            5 => {
                result = p[2][1] - p[3][1];
            }
            6 => {
                // orient2d on xz of p[0],p[2],p[3]
                result = orient2d([p[0][0], p[0][2]], [p[2][0], p[2][2]], [p[3][0], p[3][2]]);
            }
            7 => {
                result = p[2][2] - p[3][2];
            }
            8 => {
                // orient2d on yz of p[0],p[2],p[3]
                result = orient2d([p[0][1], p[0][2]], [p[2][1], p[2][2]], [p[3][1], p[3][2]]);
            }
            9 => {
                // orient2d on xy of p[0],p[1],p[3]
                result = orient2d([p[0][0], p[0][1]], [p[1][0], p[1][1]], [p[3][0], p[3][1]]);
            }
            10 => {
                result = p[1][0] - p[3][0];
            }
            11 => {
                result = p[1][1] - p[3][1];
            }
            12 => {
                result = p[0][0] - p[3][0];
            }
            _ => {
                // depth 13 (default)
                result = 1.0;
            }
        }

        if result != 0.0 {
            break;
        }
        depth += 1;
    }

    // Sign flip at depths 1, 3, 5, 8, 10
    match depth {
        1 | 3 | 5 | 8 | 10 => { result = -result; }
        _ => {}
    }

    let det = result * pn;
    ort_to_orient(det)
}

// ─── Exact lifted predicates using expansion arithmetic ───
// Ported from gDel3D/CPU/predicates.cpp orient{1,2,3}dexact_lifted.
// These compute orient determinants where the last coordinate is replaced
// by ||p||² = x² + y² + z², using Shewchuk expansion arithmetic for exact results.

/// Compute the lifted coordinate ||p||² = p[0]² - p[1]² + p[2]² as an expansion.
/// (CUDA negates the y² term: `Two_Product(-p[1], p[1], ...)`)
/// Returns (length, expansion) in a fixed-size buffer.
fn lifted_coord(p: &[f64; 3]) -> (usize, [f64; 6]) {
    let xx = two_product(p[0], p[0]);
    let yy = two_product(-p[1], p[1]);
    let zz = two_product(p[2], p[2]);
    let temp = two_two_diff(xx[1], xx[0], yy[1], yy[0]);
    let mut result = [0.0f64; 6];
    let len = fast_expansion_sum_zeroelim(&temp, &zz, &mut result);
    (len, result)
}

/// Exact orient1d_lifted: ||pa||² - ||pb||²  (CUDA predicates.cpp:776-824)
fn orient1d_lifted(pa: [f64; 3], pb: [f64; 3], lifted: bool) -> f64 {
    if !lifted {
        return pa[0] - pb[0];
    }
    let axx = two_product(pa[0], pa[0]);
    let bxx = two_product(pb[0], pb[0]);
    let aterms = two_two_diff(axx[1], axx[0], bxx[1], bxx[0]);

    let ayy = two_product(pa[1], pa[1]);
    let byy = two_product(pb[1], pb[1]);
    let bterms = two_two_diff(ayy[1], ayy[0], byy[1], byy[0]);

    let azz = two_product(pa[2], pa[2]);
    let bzz = two_product(pb[2], pb[2]);
    let cterms = two_two_diff(azz[1], azz[0], bzz[1], bzz[0]);

    let mut v = [0.0f64; 8];
    let vlen = fast_expansion_sum_zeroelim(&aterms, &bterms, &mut v);

    let mut w = [0.0f64; 12];
    let wlen = fast_expansion_sum_zeroelim(&v[..vlen], &cterms, &mut w);

    w[wlen - 1]
}

/// Exact orient2d_lifted (CUDA predicates.cpp:1513-1588)
fn orient2d_lifted(pa: [f64; 3], pb: [f64; 3], pc: [f64; 3], lifted: bool) -> f64 {
    if !lifted {
        return orient2d([pa[0], pa[1]], [pb[0], pb[1]], [pc[0], pc[1]]);
    }

    let (palen, palift) = lifted_coord(&pa);
    let (pblen, pblift) = lifted_coord(&pb);
    let (pclen, pclift) = lifted_coord(&pc);

    let mut xy1terms = [0.0f64; 12];
    let mut xy2terms = [0.0f64; 12];

    // aterms = pblift * pa[0] - pclift * pa[0]
    let xy1len = scale_expansion_zeroelim(&pblift[..pblen], pa[0], &mut xy1terms);
    let xy2len = scale_expansion_zeroelim(&pclift[..pclen], -pa[0], &mut xy2terms);
    let mut aterms = [0.0f64; 24];
    let alen = fast_expansion_sum_zeroelim(&xy1terms[..xy1len], &xy2terms[..xy2len], &mut aterms);

    // bterms = pclift * pb[0] - palift * pb[0]
    let xy1len = scale_expansion_zeroelim(&pclift[..pclen], pb[0], &mut xy1terms);
    let xy2len = scale_expansion_zeroelim(&palift[..palen], -pb[0], &mut xy2terms);
    let mut bterms = [0.0f64; 24];
    let blen = fast_expansion_sum_zeroelim(&xy1terms[..xy1len], &xy2terms[..xy2len], &mut bterms);

    // cterms = palift * pc[0] - pblift * pc[0]
    let xy1len = scale_expansion_zeroelim(&palift[..palen], pc[0], &mut xy1terms);
    let xy2len = scale_expansion_zeroelim(&pblift[..pblen], -pc[0], &mut xy2terms);
    let mut cterms = [0.0f64; 24];
    let clen = fast_expansion_sum_zeroelim(&xy1terms[..xy1len], &xy2terms[..xy2len], &mut cterms);

    let mut v = [0.0f64; 48];
    let vlen = fast_expansion_sum_zeroelim(&aterms[..alen], &bterms[..blen], &mut v);
    let mut w = [0.0f64; 72];
    let wlen = fast_expansion_sum_zeroelim(&v[..vlen], &cterms[..clen], &mut w);

    w[wlen - 1]
}

/// Exact orient3d_lifted (CUDA predicates.cpp:1590-1713)
fn orient3d_exact_lifted(pa: [f64; 3], pb: [f64; 3], pc: [f64; 3], pd: [f64; 3]) -> f64 {
    let (palen, palift) = lifted_coord(&pa);
    let (pblen, pblift) = lifted_coord(&pb);
    let (pclen, pclift) = lifted_coord(&pc);
    let (pdlen, pdlift) = lifted_coord(&pd);

    let mut xy1terms = [0.0f64; 12];
    let mut xy2terms = [0.0f64; 12];

    // ab = pblift * pa[1] - palift * pb[1]
    let xy1len = scale_expansion_zeroelim(&pblift[..pblen], pa[1], &mut xy1terms);
    let xy2len = scale_expansion_zeroelim(&palift[..palen], -pb[1], &mut xy2terms);
    let mut ab = [0.0f64; 24];
    let ablen = fast_expansion_sum_zeroelim(&xy1terms[..xy1len], &xy2terms[..xy2len], &mut ab);

    // bc = pclift * pb[1] - pblift * pc[1]
    let xy1len = scale_expansion_zeroelim(&pclift[..pclen], pb[1], &mut xy1terms);
    let xy2len = scale_expansion_zeroelim(&pblift[..pblen], -pc[1], &mut xy2terms);
    let mut bc = [0.0f64; 24];
    let bclen = fast_expansion_sum_zeroelim(&xy1terms[..xy1len], &xy2terms[..xy2len], &mut bc);

    // cd = pdlift * pc[1] - pclift * pd[1]
    let xy1len = scale_expansion_zeroelim(&pdlift[..pdlen], pc[1], &mut xy1terms);
    let xy2len = scale_expansion_zeroelim(&pclift[..pclen], -pd[1], &mut xy2terms);
    let mut cd = [0.0f64; 24];
    let cdlen = fast_expansion_sum_zeroelim(&xy1terms[..xy1len], &xy2terms[..xy2len], &mut cd);

    // da = palift * pd[1] - pdlift * pa[1]
    let xy1len = scale_expansion_zeroelim(&palift[..palen], pd[1], &mut xy1terms);
    let xy2len = scale_expansion_zeroelim(&pdlift[..pdlen], -pa[1], &mut xy2terms);
    let mut da = [0.0f64; 24];
    let dalen = fast_expansion_sum_zeroelim(&xy1terms[..xy1len], &xy2terms[..xy2len], &mut da);

    // ac = pclift * pa[1] - palift * pc[1]
    let xy1len = scale_expansion_zeroelim(&pclift[..pclen], pa[1], &mut xy1terms);
    let xy2len = scale_expansion_zeroelim(&palift[..palen], -pc[1], &mut xy2terms);
    let mut ac = [0.0f64; 24];
    let aclen = fast_expansion_sum_zeroelim(&xy1terms[..xy1len], &xy2terms[..xy2len], &mut ac);

    // bd = pdlift * pb[1] - pblift * pd[1]
    let xy1len = scale_expansion_zeroelim(&pdlift[..pdlen], pb[1], &mut xy1terms);
    let xy2len = scale_expansion_zeroelim(&pblift[..pblen], -pd[1], &mut xy2terms);
    let mut bd = [0.0f64; 24];
    let bdlen = fast_expansion_sum_zeroelim(&xy1terms[..xy1len], &xy2terms[..xy2len], &mut bd);

    // cda = cd + da + ac
    let mut temp48 = [0.0f64; 48];
    let templen = fast_expansion_sum_zeroelim(&cd[..cdlen], &da[..dalen], &mut temp48);
    let mut cda = [0.0f64; 72];
    let cdalen = fast_expansion_sum_zeroelim(&temp48[..templen], &ac[..aclen], &mut cda);

    // dab = da + ab + bd
    let templen = fast_expansion_sum_zeroelim(&da[..dalen], &ab[..ablen], &mut temp48);
    let mut dab = [0.0f64; 72];
    let dablen = fast_expansion_sum_zeroelim(&temp48[..templen], &bd[..bdlen], &mut dab);

    // Negate bd and ac for abc and bcd
    for i in 0..bdlen {
        bd[i] = -bd[i];
    }
    for i in 0..aclen {
        ac[i] = -ac[i];
    }

    // abc = ab + bc + (-ac)
    let templen = fast_expansion_sum_zeroelim(&ab[..ablen], &bc[..bclen], &mut temp48);
    let mut abc = [0.0f64; 72];
    let abclen = fast_expansion_sum_zeroelim(&temp48[..templen], &ac[..aclen], &mut abc);

    // bcd = bc + cd + (-bd)
    let templen = fast_expansion_sum_zeroelim(&bc[..bclen], &cd[..cdlen], &mut temp48);
    let mut bcd = [0.0f64; 72];
    let bcdlen = fast_expansion_sum_zeroelim(&temp48[..templen], &bd[..bdlen], &mut bcd);

    // Final: adet = bcd * pa[0], bdet = cda * (-pb[0]), cdet = dab * pc[0], ddet = abc * (-pd[0])
    let mut adet = [0.0f64; 144];
    let alen = scale_expansion_zeroelim(&bcd[..bcdlen], pa[0], &mut adet);
    let mut bdet = [0.0f64; 144];
    let blen = scale_expansion_zeroelim(&cda[..cdalen], -pb[0], &mut bdet);
    let mut cdet = [0.0f64; 144];
    let clen = scale_expansion_zeroelim(&dab[..dablen], pc[0], &mut cdet);
    let mut ddet = [0.0f64; 144];
    let dlen = scale_expansion_zeroelim(&abc[..abclen], -pd[0], &mut ddet);

    let mut abdet = [0.0f64; 288];
    let ablen2 = fast_expansion_sum_zeroelim(&adet[..alen], &bdet[..blen], &mut abdet);
    let mut cddet = [0.0f64; 288];
    let cdlen2 = fast_expansion_sum_zeroelim(&cdet[..clen], &ddet[..dlen], &mut cddet);
    let mut deter = [0.0f64; 576];
    let deterlen = fast_expansion_sum_zeroelim(&abdet[..ablen2], &cddet[..cdlen2], &mut deter);

    deter[deterlen - 1]
}

/// Adaptive orient3d_lifted with fast path + exact fallback.
/// (CUDA predicates.cpp:1715-1773)
fn orient3d_lifted(pa: [f64; 3], pb: [f64; 3], pc: [f64; 3], pd: [f64; 3], lifted: bool) -> f64 {
    if !lifted {
        return orient3d(pa, pb, pc, pd); // exact via crate
    }

    // Fast path: naive f64 computation with error bound
    let adx = pa[0] - pd[0];
    let bdx = pb[0] - pd[0];
    let cdx = pc[0] - pd[0];
    let ady = pa[1] - pd[1];
    let bdy = pb[1] - pd[1];
    let cdy = pc[1] - pd[1];
    let adz_raw = pa[2] - pd[2];
    let bdz_raw = pb[2] - pd[2];
    let cdz_raw = pc[2] - pd[2];

    let adz = adx * adx + ady * ady + adz_raw * adz_raw;
    let bdz = bdx * bdx + bdy * bdy + bdz_raw * bdz_raw;
    let cdz = cdx * cdx + cdy * cdy + cdz_raw * cdz_raw;

    let bdxcdy = bdx * cdy;
    let cdxbdy = cdx * bdy;
    let cdxady = cdx * ady;
    let adxcdy = adx * cdy;
    let adxbdy = adx * bdy;
    let bdxady = bdx * ady;

    let det = adz * (bdxcdy - cdxbdy)
        + bdz * (cdxady - adxcdy)
        + cdz * (adxbdy - bdxady);

    let permanent = (bdxcdy.abs() + cdxbdy.abs()) * adz.abs()
        + (cdxady.abs() + adxcdy.abs()) * bdz.abs()
        + (adxbdy.abs() + bdxady.abs()) * cdz.abs();

    // o3derrboundlifted = (11 + 112 * epsilon) * epsilon
    const EPSILON: f64 = 1.1102230246251565e-16; // 2^-53
    const O3DERRBOUNDLIFTED: f64 = (11.0 + 112.0 * EPSILON) * EPSILON;
    let errbound = O3DERRBOUNDLIFTED * permanent;

    if det > errbound || -det > errbound {
        return det;
    }

    orient3d_exact_lifted(pa, pb, pc, pd)
}

/// Orient4D with full 49-level SoS tie-breaking.
///
/// Determines the orientation of point `e` with respect to the circumsphere
/// of tetrahedron `(a, b, c, d)` using gDel3D's convention.
///
/// **gDel3D convention** (CommonTypes.h:95-103, PredWrapper.cpp:370-373):
///   orient4d = sphToOrient(insphere) = sign(insphere):
///   - OrientPos (+1) = point OUTSIDE circumsphere → no violation
///   - OrientNeg (-1) = point INSIDE circumsphere → violation / beneath
///
/// **Ported from:** gDel3D/CPU/PredWrapper.cpp doOrient4DAdaptSoS() + doOrientation4SoSOnly()
///
/// Never returns 0 (SoS guarantees non-degeneracy).
pub fn orient4d(
    a: [f64; 3],
    b: [f64; 3],
    c: [f64; 3],
    d: [f64; 3],
    e: [f64; 3],
    a_idx: u32,
    b_idx: u32,
    c_idx: u32,
    d_idx: u32,
    e_idx: u32,
) -> i32 {
    let sph = insphere(a, b, c, d, e);

    // insphere is now exact (Shewchuk adaptive) — non-zero means non-degenerate.
    if sph > 0.0 {
        return 1; // OrientPos
    }
    if sph < 0.0 {
        return -1; // OrientNeg
    }

    // Exactly zero: use SoS cascade
    orient4d_sos_only(a, b, c, d, e, a_idx, b_idx, c_idx, d_idx, e_idx)
}

/// Full 49-level SoS cascade for orient4d.
/// Ported from CUDA PredWrapper.cpp doOrientation4SoSOnly (lines 402-850).
fn orient4d_sos_only(
    p0: [f64; 3], p1: [f64; 3], p2: [f64; 3], p3: [f64; 3], p4: [f64; 3],
    pi0: u32, pi1: u32, pi2: u32, pi3: u32, pi4: u32,
) -> i32 {
    // Sort indices by value (bubble sort matching CUDA)
    let mut idx = [pi0, pi1, pi2, pi3, pi4];
    let mut ord: [usize; 5] = [0, 1, 2, 3, 4];
    let mut swap_count = 0u32;

    for i in 0..4 {
        for j in (i + 1..5).rev() {
            if idx[j] < idx[j - 1] {
                idx.swap(j, j - 1);
                ord.swap(j, j - 1);
                swap_count += 1;
            }
        }
    }

    // Sort points in sorted index order
    let pt4 = [p0, p1, p2, p3, p4];
    let ip = [pt4[ord[0]], pt4[ord[1]], pt4[ord[2]], pt4[ord[3]], pt4[ord[4]]];

    let mut det: f64 = 0.0;
    let mut depth: u32 = 0;

    // All sub-determinants are now exact (expansion arithmetic for lifted,
    // exact orient3d/orient2d for non-lifted). No epsilon needed.

    while det == 0.0 {
        depth += 1;

        // Set up operands and compute determinant for each depth
        // Each depth sets up op[0..3][0..2] from ip[], then evaluates
        // an orient3d_lifted, orient2d_lifted, orient1d_lifted, or constant.
        let raw = match depth {
            1 => orient3d_lifted(
                [ip[1][0], ip[1][1], ip[1][2]], [ip[2][0], ip[2][1], ip[2][2]],
                [ip[3][0], ip[3][1], ip[3][2]], [ip[4][0], ip[4][1], ip[4][2]], false),
            2 => orient3d_lifted(
                [ip[1][0], ip[1][1], ip[1][2]], [ip[2][0], ip[2][1], ip[2][2]],
                [ip[3][0], ip[3][1], ip[3][2]], [ip[4][0], ip[4][1], ip[4][2]], true),
            3 => orient3d_lifted(
                [ip[1][0], ip[1][2], ip[1][1]], [ip[2][0], ip[2][2], ip[2][1]],
                [ip[3][0], ip[3][2], ip[3][1]], [ip[4][0], ip[4][2], ip[4][1]], true),
            4 => orient3d_lifted(
                [ip[1][1], ip[1][2], ip[1][0]], [ip[2][1], ip[2][2], ip[2][0]],
                [ip[3][1], ip[3][2], ip[3][0]], [ip[4][1], ip[4][2], ip[4][0]], true),
            5 => orient3d_lifted(
                [ip[0][0], ip[0][1], ip[0][2]], [ip[2][0], ip[2][1], ip[2][2]],
                [ip[3][0], ip[3][1], ip[3][2]], [ip[4][0], ip[4][1], ip[4][2]], false),
            6 => orient2d_lifted(
                [ip[2][0], ip[2][1], ip[2][2]], [ip[3][0], ip[3][1], ip[3][2]],
                [ip[4][0], ip[4][1], ip[4][2]], false),
            7 => orient2d_lifted(
                [ip[2][0], ip[2][2], ip[2][1]], [ip[3][0], ip[3][2], ip[3][1]],
                [ip[4][0], ip[4][2], ip[4][1]], false),
            8 => orient2d_lifted(
                [ip[2][1], ip[2][2], ip[2][0]], [ip[3][1], ip[3][2], ip[3][0]],
                [ip[4][1], ip[4][2], ip[4][0]], false),
            9 => orient3d_lifted(
                [ip[0][0], ip[0][1], ip[0][2]], [ip[2][0], ip[2][1], ip[2][2]],
                [ip[3][0], ip[3][1], ip[3][2]], [ip[4][0], ip[4][1], ip[4][2]], true),
            10 => orient2d_lifted(
                [ip[2][0], ip[2][1], ip[2][2]], [ip[3][0], ip[3][1], ip[3][2]],
                [ip[4][0], ip[4][1], ip[4][2]], true),
            11 => orient2d_lifted(
                [ip[2][1], ip[2][0], ip[2][2]], [ip[3][1], ip[3][0], ip[3][2]],
                [ip[4][1], ip[4][0], ip[4][2]], true),
            12 => orient3d_lifted(
                [ip[0][0], ip[0][2], ip[0][1]], [ip[2][0], ip[2][2], ip[2][1]],
                [ip[3][0], ip[3][2], ip[3][1]], [ip[4][0], ip[4][2], ip[4][1]], true),
            13 => orient2d_lifted(
                [ip[2][2], ip[2][0], ip[2][1]], [ip[3][2], ip[3][0], ip[3][1]],
                [ip[4][2], ip[4][0], ip[4][1]], true),
            14 => orient3d_lifted(
                [ip[0][1], ip[0][2], ip[0][0]], [ip[2][1], ip[2][2], ip[2][0]],
                [ip[3][1], ip[3][2], ip[3][0]], [ip[4][1], ip[4][2], ip[4][0]], true),
            15 => orient3d_lifted(
                [ip[0][0], ip[0][1], ip[0][2]], [ip[1][0], ip[1][1], ip[1][2]],
                [ip[3][0], ip[3][1], ip[3][2]], [ip[4][0], ip[4][1], ip[4][2]], false),
            16 => orient2d_lifted(
                [ip[1][0], ip[1][1], ip[1][2]], [ip[3][0], ip[3][1], ip[3][2]],
                [ip[4][0], ip[4][1], ip[4][2]], false),
            17 => orient2d_lifted(
                [ip[1][0], ip[1][2], ip[1][1]], [ip[3][0], ip[3][2], ip[3][1]],
                [ip[4][0], ip[4][2], ip[4][1]], false),
            18 => orient2d_lifted(
                [ip[1][1], ip[1][2], ip[1][0]], [ip[3][1], ip[3][2], ip[3][0]],
                [ip[4][1], ip[4][2], ip[4][0]], false),
            19 => orient2d_lifted(
                [ip[0][0], ip[0][1], ip[0][2]], [ip[3][0], ip[3][1], ip[3][2]],
                [ip[4][0], ip[4][1], ip[4][2]], false),
            20 => orient1d_lifted(
                [ip[3][0], ip[3][1], ip[3][2]], [ip[4][0], ip[4][1], ip[4][2]], false),
            21 => orient1d_lifted(
                [ip[3][1], ip[3][0], ip[3][2]], [ip[4][1], ip[4][0], ip[4][2]], false),
            22 => orient2d_lifted(
                [ip[0][0], ip[0][2], ip[0][1]], [ip[3][0], ip[3][2], ip[3][1]],
                [ip[4][0], ip[4][2], ip[4][1]], false),
            23 => orient1d_lifted(
                [ip[3][2], ip[3][0], ip[3][1]], [ip[4][2], ip[4][0], ip[4][1]], false),
            24 => orient2d_lifted(
                [ip[0][1], ip[0][2], ip[0][0]], [ip[3][1], ip[3][2], ip[3][0]],
                [ip[4][1], ip[4][2], ip[4][0]], false),
            25 => orient3d_lifted(
                [ip[0][0], ip[0][1], ip[0][2]], [ip[1][0], ip[1][1], ip[1][2]],
                [ip[3][0], ip[3][1], ip[3][2]], [ip[4][0], ip[4][1], ip[4][2]], true),
            26 => orient2d_lifted(
                [ip[1][0], ip[1][1], ip[1][2]], [ip[3][0], ip[3][1], ip[3][2]],
                [ip[4][0], ip[4][1], ip[4][2]], true),
            27 => orient2d_lifted(
                [ip[1][1], ip[1][0], ip[1][2]], [ip[3][1], ip[3][0], ip[3][2]],
                [ip[4][1], ip[4][0], ip[4][2]], true),
            28 => orient2d_lifted(
                [ip[0][0], ip[0][1], ip[0][2]], [ip[3][0], ip[3][1], ip[3][2]],
                [ip[4][0], ip[4][1], ip[4][2]], true),
            29 => orient1d_lifted(
                [ip[3][0], ip[3][1], ip[3][2]], [ip[4][0], ip[4][1], ip[4][2]], true),
            30 => orient2d_lifted(
                [ip[0][1], ip[0][0], ip[0][2]], [ip[3][1], ip[3][0], ip[3][2]],
                [ip[4][1], ip[4][0], ip[4][2]], true),
            31 => orient3d_lifted(
                [ip[0][0], ip[0][2], ip[0][1]], [ip[1][0], ip[1][2], ip[1][1]],
                [ip[3][0], ip[3][2], ip[3][1]], [ip[4][0], ip[4][2], ip[4][1]], true),
            32 => orient2d_lifted(
                [ip[1][2], ip[1][0], ip[1][1]], [ip[3][2], ip[3][0], ip[3][1]],
                [ip[4][2], ip[4][0], ip[4][1]], true),
            33 => orient2d_lifted(
                [ip[0][2], ip[0][0], ip[0][1]], [ip[3][2], ip[3][0], ip[3][1]],
                [ip[4][2], ip[4][0], ip[4][1]], true),
            34 => orient3d_lifted(
                [ip[0][1], ip[0][2], ip[0][0]], [ip[1][1], ip[1][2], ip[1][0]],
                [ip[3][1], ip[3][2], ip[3][0]], [ip[4][1], ip[4][2], ip[4][0]], true),
            35 => orient3d_lifted(
                [ip[0][0], ip[0][1], ip[0][2]], [ip[1][0], ip[1][1], ip[1][2]],
                [ip[2][0], ip[2][1], ip[2][2]], [ip[4][0], ip[4][1], ip[4][2]], false),
            36 => orient2d_lifted(
                [ip[1][0], ip[1][1], ip[1][2]], [ip[2][0], ip[2][1], ip[2][2]],
                [ip[4][0], ip[4][1], ip[4][2]], false),
            37 => orient2d_lifted(
                [ip[1][0], ip[1][2], ip[1][1]], [ip[2][0], ip[2][2], ip[2][1]],
                [ip[4][0], ip[4][2], ip[4][1]], false),
            38 => orient2d_lifted(
                [ip[1][1], ip[1][2], ip[1][0]], [ip[2][1], ip[2][2], ip[2][0]],
                [ip[4][1], ip[4][2], ip[4][0]], false),
            39 => orient2d_lifted(
                [ip[0][0], ip[0][1], ip[0][2]], [ip[2][0], ip[2][1], ip[2][2]],
                [ip[4][0], ip[4][1], ip[4][2]], false),
            40 => orient1d_lifted(
                [ip[2][0], ip[2][1], ip[2][2]], [ip[4][0], ip[4][1], ip[4][2]], false),
            41 => orient1d_lifted(
                [ip[2][1], ip[2][0], ip[2][2]], [ip[4][1], ip[4][0], ip[4][2]], false),
            42 => orient2d_lifted(
                [ip[0][0], ip[0][2], ip[0][1]], [ip[2][0], ip[2][2], ip[2][1]],
                [ip[4][0], ip[4][2], ip[4][1]], false),
            43 => orient1d_lifted(
                [ip[2][2], ip[2][0], ip[2][1]], [ip[4][2], ip[4][0], ip[4][1]], false),
            44 => orient2d_lifted(
                [ip[0][1], ip[0][2], ip[0][0]], [ip[2][1], ip[2][2], ip[2][0]],
                [ip[4][1], ip[4][2], ip[4][0]], false),
            45 => orient2d_lifted(
                [ip[0][0], ip[0][1], ip[0][2]], [ip[1][0], ip[1][1], ip[1][2]],
                [ip[4][0], ip[4][1], ip[4][2]], false),
            46 => orient1d_lifted(
                [ip[1][0], ip[1][1], ip[1][2]], [ip[4][0], ip[4][1], ip[4][2]], false),
            47 => orient1d_lifted(
                [ip[1][1], ip[1][0], ip[1][2]], [ip[4][1], ip[4][0], ip[4][2]], false),
            48 => orient1d_lifted(
                [ip[0][0], ip[0][1], ip[0][2]], [ip[4][0], ip[4][1], ip[4][2]], false),
            _ => {
                // depth 49: always +1.0
                1.0
            }
        };

        det = raw;

        if depth >= 49 {
            break;
        }
    }

    // Sign flip at specific depths (CUDA PredWrapper.cpp:828-840)
    match depth {
        1 | 3 | 7 | 9 | 11 | 14 | 15 | 16 | 18 | 20 | 22 | 23 | 26 | 30 | 31 | 32 | 37 | 39 | 41 | 44 | 46 => {
            det = -det;
        }
        _ => {}
    }

    // Flip for odd swap count
    if swap_count % 2 != 0 {
        det = -det;
    }

    sph_to_orient(det)
}

/// Orient4D without SoS (for testing or when indices are not available).
/// Returns the raw sign of the insphere test.
pub fn orient4d_no_sos(
    a: [f64; 3],
    b: [f64; 3],
    c: [f64; 3],
    d: [f64; 3],
    e: [f64; 3],
) -> i32 {
    sign(insphere(a, b, c, d, e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orient3d_positive() {
        // Right-hand rule: a,b,c counter-clockwise from above, d below
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        let c = [0.0, 1.0, 0.0];
        let d = [0.0, 0.0, 1.0];
        assert!(orient3d(a, b, c, d) < 0.0); // d is above the plane
    }

    #[test]
    fn test_orient3d_coplanar() {
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        let c = [0.0, 1.0, 0.0];
        let d = [1.0, 1.0, 0.0];
        assert_eq!(orient3d(a, b, c, d), 0.0);
    }

    #[test]
    fn test_insphere_inside() {
        // Regular tet, point at center should be inside
        let a = [1.0, 1.0, 1.0];
        let b = [1.0, -1.0, -1.0];
        let c = [-1.0, 1.0, -1.0];
        let d = [-1.0, -1.0, 1.0];
        let e = [0.0, 0.0, 0.0]; // center
        // orient3d(a,b,c,d) needs to be positive for insphere sign to be meaningful
        let o = orient3d(a, b, c, d);
        let s = insphere(a, b, c, d, e);
        // If orient is negative, flip sign of insphere
        let inside = if o > 0.0 { s > 0.0 } else { s < 0.0 };
        assert!(inside, "Center should be inside circumsphere");
    }

    #[test]
    fn test_insphere_outside() {
        let a = [1.0, 1.0, 1.0];
        let b = [1.0, -1.0, -1.0];
        let c = [-1.0, 1.0, -1.0];
        let d = [-1.0, -1.0, 1.0];
        let e = [10.0, 10.0, 10.0]; // far away
        let o = orient3d(a, b, c, d);
        let s = insphere(a, b, c, d, e);
        let inside = if o > 0.0 { s > 0.0 } else { s < 0.0 };
        assert!(!inside, "Far point should be outside circumsphere");
    }

    #[test]
    fn test_circumcenter() {
        // Regular tetrahedron centered at origin
        let a = [1.0, 1.0, 1.0];
        let b = [1.0, -1.0, -1.0];
        let c = [-1.0, 1.0, -1.0];
        let d = [-1.0, -1.0, 1.0];
        let cc = circumcenter(a, b, c, d).unwrap();
        for i in 0..3 {
            assert!(cc[i].abs() < 1e-10, "Circumcenter should be at origin");
        }
    }

    #[test]
    fn test_orient3d_sos_coplanar() {
        // 4 coplanar points → raw orient3d = 0, but SoS returns non-zero
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        let c = [0.0, 1.0, 0.0];
        let d = [1.0, 1.0, 0.0];
        assert_eq!(orient3d(a, b, c, d), 0.0);
        let sos = orient3d_sos(a, b, c, d, 0, 1, 2, 3);
        assert_ne!(sos, 0, "SoS should never return 0");
    }

    #[test]
    fn test_orient3d_sos_swap_parity() {
        // Swapping two vertices should flip the SoS result
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        let c = [0.0, 1.0, 0.0];
        let d = [1.0, 1.0, 0.0];
        let r1 = orient3d_sos(a, b, c, d, 0, 1, 2, 3);
        let r2 = orient3d_sos(b, a, c, d, 1, 0, 2, 3);
        assert_eq!(r1, -r2, "Swapping two vertices should flip orient3d_sos");
    }

    #[test]
    fn test_orient3d_sos_matches_nondegenerate() {
        // For non-degenerate points, SoS should match raw sign under ortToOrient
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        let c = [0.0, 1.0, 0.0];
        let d = [0.0, 0.0, 1.0];
        let raw = orient3d(a, b, c, d);
        let sos = orient3d_sos(a, b, c, d, 0, 1, 2, 3);
        let expected = ort_to_orient(raw);
        assert_eq!(sos, expected, "SoS should match raw for non-degenerate case");
    }

    #[test]
    fn test_orient4d_sos_cospherical() {
        // 5 cospherical points (on unit sphere) → raw insphere = 0, SoS non-zero
        let a = [1.0, 0.0, 0.0];
        let b = [-1.0, 0.0, 0.0];
        let c = [0.0, 1.0, 0.0];
        let d = [0.0, -1.0, 0.0];
        let e = [0.0, 0.0, 1.0];
        // Verify raw insphere is zero (all on unit sphere)
        let raw = insphere(a, b, c, d, e);
        assert_eq!(raw, 0.0, "Points on unit sphere should give insphere=0");
        let sos = orient4d(a, b, c, d, e, 0, 1, 2, 3, 4);
        assert_ne!(sos, 0, "SoS should never return 0");
    }

    #[test]
    fn test_orient4d_sos_matches_nondegenerate() {
        // For non-degenerate points, SoS should match raw insphere sign
        let a = [1.0, 1.0, 1.0];
        let b = [1.0, -1.0, -1.0];
        let c = [-1.0, 1.0, -1.0];
        let d = [-1.0, -1.0, 1.0];
        let e = [0.0, 0.0, 0.0]; // inside
        let raw = insphere(a, b, c, d, e);
        assert_ne!(raw, 0.0);
        let expected = sph_to_orient(raw);
        let sos = orient4d(a, b, c, d, e, 0, 1, 2, 3, 4);
        assert_eq!(sos, expected, "SoS should match raw for non-degenerate case");
    }
}
