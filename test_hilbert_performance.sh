#!/bin/bash
# Quick performance comparison: Sequential vs Hilbert sorting

echo "================================"
echo "Testing Hilbert Curve Performance"
echo "================================"
echo ""

# Test with one of the passing tests
TEST_NAME="test_delaunay_grid_4x4x4"

echo "1. Testing with Hilbert sorting ENABLED..."
echo "   (enable_hilbert_sorting: true in types.rs)"
cargo test --release $TEST_NAME -- --nocapture 2>&1 | grep -E "\[GPU STATE\]|\[DEBUG\] After iteration|\[TIMING\] Iteration" | head -20

echo ""
echo "2. To test baseline (sequential), set enable_hilbert_sorting: false in src/types.rs"
echo ""
echo "Note: Look for these markers:"
echo "  - [GPU STATE] messages show if Hilbert sorting is active"
echo "  - [DEBUG] After iteration messages show remaining points"
echo "  - [TIMING] messages show iteration duration"
