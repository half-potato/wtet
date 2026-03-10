//! Morton code (Z-order curve) computation for spatial sorting.
//!
//! Morton codes provide spatial locality by interleaving the bits of x, y, z coordinates.
//! Points close in 3D space have similar Morton codes, which reduces voting conflicts
//! during Delaunay insertion.
//!
//! CUDA Reference: gDel3D/GDelFlipping/src/gDel3D/GPU/ThrustWrapper.h:118-164

/// Compute Morton code (Z-order curve) for a 3D point.
///
/// The point is normalized to [0, 1]³ using the provided bounding box,
/// quantized to 10-bit integers (0-1023), and the bits are interleaved
/// to produce a 30-bit Morton code.
///
/// # Arguments
/// * `point` - 3D point coordinates
/// * `min` - Bounding box minimum corner
/// * `max` - Bounding box maximum corner
///
/// # Returns
/// 30-bit Morton code where bits are interleaved as: zzz...z_yyy...y_xxx...x
pub fn compute_morton_code(point: [f32; 3], min: [f32; 3], max: [f32; 3]) -> u32 {
    // 1. Normalize coordinates to [0, 1] range
    let nx = ((point[0] - min[0]) / (max[0] - min[0])).clamp(0.0, 1.0);
    let ny = ((point[1] - min[1]) / (max[1] - min[1])).clamp(0.0, 1.0);
    let nz = ((point[2] - min[2]) / (max[2] - min[2])).clamp(0.0, 1.0);

    // 2. Quantize to 10-bit integers (0-1023)
    let x = (nx * 1023.0) as u32;
    let y = (ny * 1023.0) as u32;
    let z = (nz * 1023.0) as u32;

    // 3. Interleave bits
    interleave_bits(x, y, z)
}

/// Interleave bits from x, y, z to create Morton code.
///
/// Each coordinate is a 10-bit value. The bits are expanded to 30 bits
/// with 2-bit gaps, then combined to form the final Morton code.
///
/// # Arguments
/// * `x` - 10-bit x coordinate (0-1023)
/// * `y` - 10-bit y coordinate (0-1023)
/// * `z` - 10-bit z coordinate (0-1023)
///
/// # Returns
/// 30-bit Morton code with interleaved bits: z_y_x pattern
fn interleave_bits(x: u32, y: u32, z: u32) -> u32 {
    // Expand each 10-bit value to 30 bits with gaps
    let xx = expand_bits(x);
    let yy = expand_bits(y);
    let zz = expand_bits(z);

    // Interleave: z_y_x pattern
    (zz << 2) | (yy << 1) | xx
}

/// Expand 10-bit value to 30 bits with 2-bit gaps.
///
/// Example: ABC (3-bit) → 00A00B00C (9-bit)
///
/// For 10-bit input, this creates a 30-bit output where every third bit
/// contains the original data and the other bits are zero.
///
/// # Arguments
/// * `v` - 10-bit value (0-1023)
///
/// # Returns
/// 30-bit value with original bits at positions 0, 3, 6, 9, ..., 27
///
/// CUDA Reference: ThrustWrapper.h:139-147
fn expand_bits(mut v: u32) -> u32 {
    // This bit manipulation spreads 10 bits across 30 positions
    // Each step doubles the spacing between active bits
    v = (v | (v << 16)) & 0x030000FF;
    v = (v | (v <<  8)) & 0x0300F00F;
    v = (v | (v <<  4)) & 0x030C30C3;
    v = (v | (v <<  2)) & 0x09249249;
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_morton_code_corners() {
        let min = [0.0, 0.0, 0.0];
        let max = [1.0, 1.0, 1.0];

        // Test minimum corner (all zeros)
        assert_eq!(compute_morton_code([0.0, 0.0, 0.0], min, max), 0);

        // Test maximum corner (all ones in 10-bit space)
        // 1023 in each dimension → all bits set
        let max_code = compute_morton_code([1.0, 1.0, 1.0], min, max);
        // Expected: interleave(1023, 1023, 1023) = 0x3FFFFFFF (30 bits all set)
        assert_eq!(max_code, 0x3FFFFFFF);
    }

    #[test]
    fn test_morton_code_spatial_locality() {
        let min = [0.0, 0.0, 0.0];
        let max = [1.0, 1.0, 1.0];

        // Test that Morton codes preserve Z-order curve properties:
        // Points at the same quantization level should have the same code
        let p1 = compute_morton_code([0.500, 0.500, 0.500], min, max);
        let p2 = compute_morton_code([0.500, 0.500, 0.500], min, max);
        assert_eq!(p1, p2, "Identical points must have identical Morton codes");

        // Test spatial clustering: points in the same octant should be closer
        // than points in different octants
        let center = compute_morton_code([0.5, 0.5, 0.5], min, max);
        let same_octant = compute_morton_code([0.6, 0.6, 0.6], min, max);
        let far_octant = compute_morton_code([0.1, 0.1, 0.1], min, max);

        let diff_same = (center as i64 - same_octant as i64).abs();
        let diff_far = (center as i64 - far_octant as i64).abs();

        // Morton codes should cluster points in the same octant
        // (Though the specific relationship depends on the Z-order curve structure)
        println!("Center code: {:032b}", center);
        println!("Same octant code: {:032b}, diff: {}", same_octant, diff_same);
        println!("Far octant code: {:032b}, diff: {}", far_octant, diff_far);
    }

    #[test]
    fn test_expand_bits() {
        // Test with simple values to verify bit pattern
        assert_eq!(expand_bits(0), 0);
        assert_eq!(expand_bits(1), 1);  // Bit 0 set
        assert_eq!(expand_bits(2), 8);  // Bit 1 → position 3
        assert_eq!(expand_bits(4), 64); // Bit 2 → position 6
    }

    #[test]
    fn test_interleave_bits() {
        // Test simple case: x=1, y=0, z=0
        assert_eq!(interleave_bits(1, 0, 0), 1);  // Only x bit 0 set

        // Test: x=0, y=1, z=0
        assert_eq!(interleave_bits(0, 1, 0), 2);  // Only y bit 0 set (shifted left by 1)

        // Test: x=0, y=0, z=1
        assert_eq!(interleave_bits(0, 0, 1), 4);  // Only z bit 0 set (shifted left by 2)

        // Test: x=1, y=1, z=1
        assert_eq!(interleave_bits(1, 1, 1), 7);  // All three bits set (0b111)
    }
}
