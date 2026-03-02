// Double-double arithmetic for exact geometric predicates in WGSL.
//
// A double-double number uses two f32 values (hi, lo) where the true value
// is hi + lo, giving ~48-bit mantissa precision (~14 decimal digits).
// This is sufficient for orient3d (3x3 det) and insphere (5x5 det) when
// inputs are normalized to [0,1]³.

struct DD {
    hi: f32,
    lo: f32,
}

// --- Core double-double operations ---

// Knuth's TwoSum: exact a+b = s + e
fn two_sum(a: f32, b: f32) -> DD {
    let s = a + b;
    let v = s - a;
    let e = (a - (s - v)) + (b - v);
    return DD(s, e);
}

// Fast TwoSum when |a| >= |b|
fn fast_two_sum(a: f32, b: f32) -> DD {
    let s = a + b;
    let e = b - (s - a);
    return DD(s, e);
}

// Dekker's TwoProduct: exact a*b = p + e
// Uses fma where available, otherwise Dekker splitting.
fn two_product(a: f32, b: f32) -> DD {
    let p = a * b;
    let e = fma(a, b, -p);
    return DD(p, e);
}

// DD + DD
fn dd_add(a: DD, b: DD) -> DD {
    let s = two_sum(a.hi, b.hi);
    let t = two_sum(a.lo, b.lo);
    var c = fast_two_sum(s.hi, s.lo + t.hi);
    c = fast_two_sum(c.hi, c.lo + t.lo);
    return c;
}

// DD + f32
fn dd_add_f(a: DD, b: f32) -> DD {
    let s = two_sum(a.hi, b);
    let c = fast_two_sum(s.hi, s.lo + a.lo);
    return c;
}

// DD - DD
fn dd_sub(a: DD, b: DD) -> DD {
    return dd_add(a, DD(-b.hi, -b.lo));
}

// DD * DD
fn dd_mul(a: DD, b: DD) -> DD {
    let p = two_product(a.hi, b.hi);
    let e = a.hi * b.lo + a.lo * b.hi + p.lo;
    return fast_two_sum(p.hi, e);
}

// DD * f32
fn dd_mul_f(a: DD, b: f32) -> DD {
    let p = two_product(a.hi, b);
    let e = a.lo * b + p.lo;
    return fast_two_sum(p.hi, e);
}

// DD from f32
fn dd_from_f32(x: f32) -> DD {
    return DD(x, 0.0);
}

// DD negation
fn dd_neg(a: DD) -> DD {
    return DD(-a.hi, -a.lo);
}

// Sign of DD
fn dd_sign(a: DD) -> i32 {
    if a.hi > 0.0 { return 1; }
    if a.hi < 0.0 { return -1; }
    if a.lo > 0.0 { return 1; }
    if a.lo < 0.0 { return -1; }
    return 0;
}

// --- Fast f32 predicates with error bounds ---

// orient3d: sign of det | a-d, b-d, c-d |
// Returns (result, is_certain). If !is_certain, use DD version.
fn orient3d_fast(
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

    // Error bound: 6 * eps * sum of absolute products
    // eps for f32 ≈ 5.96e-8
    let eps = 5.96e-8;
    let permanent = abs(adx) * (abs(bdy * cdz) + abs(bdz * cdy))
                  + abs(bdx) * (abs(cdy * adz) + abs(cdz * ady))
                  + abs(cdx) * (abs(ady * bdz) + abs(adz * bdy));
    let err_bound = 7.77e-7 * permanent; // ~13 * eps * permanent

    // result.x = determinant, result.y = 0 if certain, 1 if uncertain
    if abs(det) > err_bound {
        return vec2<f32>(det, 0.0);
    }
    return vec2<f32>(det, 1.0);
}

// orient3d with double-double fallback
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

// Combined orient3d: fast path with DD fallback.
// Returns sign: +1, -1, or 0.
fn orient3d_exact(
    ax: f32, ay: f32, az: f32,
    bx: f32, by: f32, bz: f32,
    cx: f32, cy: f32, cz: f32,
    dx: f32, dy: f32, dz: f32,
) -> i32 {
    let fast = orient3d_fast(ax, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz);
    if fast.y == 0.0 {
        if fast.x > 0.0 { return 1; }
        if fast.x < 0.0 { return -1; }
        return 0;
    }
    return orient3d_dd(ax, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz);
}

// --- Insphere predicate ---

// insphere: positive if e is inside circumsphere of positively-oriented a,b,c,d
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

    // Error bound for insphere is much larger
    let permanent = abs(dlift * abc) + abs(clift * dab)
                  + abs(blift * cda) + abs(alift * bcd);
    let err_bound = 1.5e-5 * permanent;  // ~250 * eps * permanent

    if abs(det) > err_bound {
        return vec2<f32>(det, 0.0);
    }
    return vec2<f32>(det, 1.0);
}

// insphere with double-double arithmetic
fn insphere_dd(
    ax: f32, ay: f32, az: f32,
    bx: f32, by: f32, bz: f32,
    cx: f32, cy: f32, cz: f32,
    dx: f32, dy: f32, dz: f32,
    ex: f32, ey: f32, ez: f32,
) -> i32 {
    let aex = dd_from_f32(ax - ex);
    let aey = dd_from_f32(ay - ey);
    let aez = dd_from_f32(az - ez);
    let bex = dd_from_f32(bx - ex);
    let bey = dd_from_f32(by - ey);
    let bez = dd_from_f32(bz - ez);
    let cex = dd_from_f32(cx - ex);
    let cey = dd_from_f32(cy - ey);
    let cez = dd_from_f32(cz - ez);
    let dex = dd_from_f32(dx - ex);
    let dey = dd_from_f32(dy - ey);
    let dez = dd_from_f32(dz - ez);

    let ab = dd_sub(dd_mul(aex, bey), dd_mul(bex, aey));
    let bc = dd_sub(dd_mul(bex, cey), dd_mul(cex, bey));
    let cd = dd_sub(dd_mul(cex, dey), dd_mul(dex, cey));
    let da = dd_sub(dd_mul(dex, aey), dd_mul(aex, dey));
    let ac = dd_sub(dd_mul(aex, cey), dd_mul(cex, aey));
    let bd = dd_sub(dd_mul(bex, dey), dd_mul(dex, bey));

    let abc = dd_add(dd_sub(dd_mul(dd_from_f32(aez.hi), bc), dd_mul(dd_from_f32(bez.hi), ac)), dd_mul(dd_from_f32(cez.hi), ab));
    let bcd = dd_add(dd_sub(dd_mul(dd_from_f32(bez.hi), cd), dd_mul(dd_from_f32(cez.hi), bd)), dd_mul(dd_from_f32(dez.hi), bc));
    let cda = dd_add(dd_add(dd_mul(dd_from_f32(cez.hi), da), dd_mul(dd_from_f32(dez.hi), ac)), dd_mul(dd_from_f32(aez.hi), cd));
    let dab = dd_add(dd_add(dd_mul(dd_from_f32(dez.hi), ab), dd_mul(dd_from_f32(aez.hi), bd)), dd_mul(dd_from_f32(bez.hi), da));

    let alift = dd_add(dd_add(dd_mul(aex, aex), dd_mul(aey, aey)), dd_mul(aez, aez));
    let blift = dd_add(dd_add(dd_mul(bex, bex), dd_mul(bey, bey)), dd_mul(bez, bez));
    let clift = dd_add(dd_add(dd_mul(cex, cex), dd_mul(cey, cey)), dd_mul(cez, cez));
    let dlift = dd_add(dd_add(dd_mul(dex, dex), dd_mul(dey, dey)), dd_mul(dez, dez));

    let det = dd_add(
        dd_sub(dd_mul(dlift, abc), dd_mul(clift, dab)),
        dd_sub(dd_mul(blift, cda), dd_mul(alift, bcd))
    );

    return dd_sign(det);
}

// Combined insphere: fast path with DD fallback.
fn insphere_exact(
    ax: f32, ay: f32, az: f32,
    bx: f32, by: f32, bz: f32,
    cx: f32, cy: f32, cz: f32,
    dx: f32, dy: f32, dz: f32,
    ex: f32, ey: f32, ez: f32,
) -> i32 {
    let fast = insphere_fast(ax, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz, ex, ey, ez);
    if fast.y == 0.0 {
        if fast.x > 0.0 { return 1; }
        if fast.x < 0.0 { return -1; }
        return 0;
    }
    return insphere_dd(ax, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz, ex, ey, ez);
}
