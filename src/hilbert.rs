//! Hilbert curve computation using the `hilbert_index` crate.
//!
//! Provides spatial ordering for point insertion to reduce voting conflicts
//! during Delaunay triangulation.

use hilbert_index::ToHilbertIndex;

/// Compute Hilbert curve index for a 3D point.
///
/// Uses 8-bit quantization per axis (256 subdivisions) to produce
/// a Hilbert index via the `hilbert_index` crate (Butz's algorithm).
///
/// # Arguments
/// * `point` - 3D point coordinates
/// * `min` - Bounding box minimum corner
/// * `max` - Bounding box maximum corner
///
/// # Returns
/// usize Hilbert index (level 8: 24 bits total for 3D)
pub fn compute_hilbert_index(point: [f32; 3], min: [f32; 3], max: [f32; 3]) -> usize {
    // 1. Normalize to [0, 1]³
    let nx = ((point[0] - min[0]) / (max[0] - min[0])).clamp(0.0, 1.0);
    let ny = ((point[1] - min[1]) / (max[1] - min[1])).clamp(0.0, 1.0);
    let nz = ((point[2] - min[2]) / (max[2] - min[2])).clamp(0.0, 1.0);

    // 2. Quantize to integers based on Hilbert curve level
    // The crate uses level as recursion depth, where coordinates must be in [0, 2^level)
    // Testing level 8: 2^8 = 256 subdivisions per axis
    const LEVEL: usize = 8;
    let max_coord = (1 << LEVEL) - 1; // 255 for level 8
    let x = (nx * max_coord as f32).round() as usize;
    let y = (ny * max_coord as f32).round() as usize;
    let z = (nz * max_coord as f32).round() as usize;

    // 3. Compute Hilbert index using established algorithm
    // Level 8: 2^8 = 256 subdivisions per axis, 3*8 = 24 bits total
    [x, y, z].to_hilbert_index(LEVEL)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_hilbert_index_corners() {
        let min = [0.0, 0.0, 0.0];
        let max = [1.0, 1.0, 1.0];

        // Test all 8 corners of the unit cube
        let corners = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ];

        let indices: Vec<_> = corners
            .iter()
            .map(|&p| compute_hilbert_index(p, min, max))
            .collect();

        // All indices should be unique
        let unique_count = indices.iter().collect::<HashSet<_>>().len();
        assert_eq!(
            unique_count, 8,
            "Corner indices should be unique: {:?}",
            indices
        );
    }

    #[test]
    fn test_hilbert_locality() {
        let min = [0.0, 0.0, 0.0];
        let max = [1.0, 1.0, 1.0];

        // Test that nearby points generally have closer indices than far points
        let test_points = [
            ([0.5, 0.5, 0.5], "center"),
            ([0.51, 0.5, 0.5], "nearby_x"),
            ([0.5, 0.51, 0.5], "nearby_y"),
            ([0.5, 0.5, 0.51], "nearby_z"),
            ([0.1, 0.1, 0.1], "far_1"),
            ([0.9, 0.9, 0.9], "far_2"),
        ];

        let indices: Vec<_> = test_points
            .iter()
            .map(|(p, _)| compute_hilbert_index(*p, min, max))
            .collect();

        // Verify uniqueness
        let unique_count = indices.iter().collect::<HashSet<_>>().len();
        assert_eq!(unique_count, test_points.len());

        // Check average locality preservation
        let center_idx = indices[0] as i64;
        let nearby_diffs: Vec<_> = indices[1..4]
            .iter()
            .map(|&idx| (center_idx - idx as i64).abs())
            .collect();
        let far_diffs: Vec<_> = indices[4..6]
            .iter()
            .map(|&idx| (center_idx - idx as i64).abs())
            .collect();

        let avg_nearby = nearby_diffs.iter().sum::<i64>() / nearby_diffs.len() as i64;
        let avg_far = far_diffs.iter().sum::<i64>() / far_diffs.len() as i64;

        // On average, nearby points should have closer indices
        assert!(
            avg_nearby < avg_far,
            "Hilbert curve should preserve locality on average: nearby={}, far={}",
            avg_nearby, avg_far
        );
    }
}
