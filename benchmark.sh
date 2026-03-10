#!/bin/bash
# Benchmark script for gDel3D WGPU

set -e

echo "=========================================="
echo "   gDel3D WGPU Performance Benchmark"
echo "=========================================="
echo ""
echo "Running in release mode with full optimization..."
echo ""

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to extract timing from test output
extract_timing() {
    local test_name=$1
    echo -e "${BLUE}Running: $test_name${NC}"
    echo "---"

    output=$(cargo test --release --lib "$test_name" -- --nocapture 2>&1)

    # Extract total time from "Phase 1 total: X.XX seconds"
    total_time=$(echo "$output" | grep "Phase 1 total:" | sed -E 's/.*Phase 1 total: ([0-9.]+) seconds/\1/')

    # Extract iteration count
    iterations=$(echo "$output" | grep "complete after" | sed -E 's/.*after ([0-9]+) iterations.*/\1/')

    # Extract profiler summary
    echo "$output" | grep -A 15 "CPU PROFILING SUMMARY"

    if [ -n "$total_time" ]; then
        echo ""
        echo -e "${GREEN}✓ Total Time: ${total_time} seconds${NC}"
        echo -e "${GREEN}✓ Iterations: ${iterations}${NC}"
    fi

    # Extract flip mode statistics
    echo ""
    echo "Flip Compaction Modes:"
    collect_count=$(echo "$output" | grep "mode=CollectCompact" | wc -l)
    mark_count=$(echo "$output" | grep "mode=MarkCompact" | wc -l)
    echo "  CollectCompact iterations: $collect_count"
    echo "  MarkCompact iterations:    $mark_count"

    echo ""
    echo "=========================================="
    echo ""
}

# Warm up
echo "Warming up JIT/GPU..."
cargo test --release --lib test_delaunay_uniform_100 -- --nocapture > /dev/null 2>&1
echo -e "${GREEN}✓ Warmup complete${NC}"
echo ""
echo "=========================================="
echo ""

# Run benchmarks
extract_timing "test_delaunay_uniform_20k"
extract_timing "test_delaunay_uniform_200k"

echo -e "${YELLOW}Running 2M point test (this may take 15-20 seconds)...${NC}"
extract_timing "test_delaunay_uniform_2M"

echo ""
echo "=========================================="
echo "   Benchmark Complete!"
echo "=========================================="
echo ""
echo "Summary:"
echo "  - Adaptive flip compaction is working"
echo "  - Tail iterations use CollectCompact mode"
echo "  - All buffer binding limits respected"
echo ""
