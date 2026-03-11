"""Benchmark gdel3d performance with different point set sizes."""

import numpy as np
import gdel3d
import time

def benchmark(num_points, num_runs=3):
    """Benchmark Delaunay triangulation for given number of points."""
    print(f"\n{'='*60}")
    print(f"Benchmark: {num_points:,} points ({num_runs} runs)")
    print(f"{'='*60}")

    # Generate random points
    np.random.seed(42)
    points = np.random.rand(num_points, 3).astype(np.float32)

    # Configuration
    config = gdel3d.Config(
        enable_flipping=True,
        enable_splaying=True,
    )

    times = []
    num_tets_list = []
    num_failed_list = []

    for run in range(num_runs):
        start = time.time()
        result = gdel3d.delaunay(points, config)
        elapsed = time.time() - start

        times.append(elapsed)
        num_tets_list.append(result.num_tets)
        num_failed_list.append(result.num_failed)

        print(f"  Run {run+1}: {elapsed*1000:.1f} ms, {result.num_tets:,} tets, {result.num_failed} failed")

    # Statistics
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    avg_tets = np.mean(num_tets_list)

    print(f"\nStatistics:")
    print(f"  Time: {avg_time*1000:.1f} ± {std_time*1000:.1f} ms")
    print(f"  Range: {min_time*1000:.1f} - {max_time*1000:.1f} ms")
    print(f"  Avg tets: {avg_tets:,.0f}")
    print(f"  Throughput: {num_points/avg_time:,.0f} points/sec")

    return avg_time, avg_tets


if __name__ == "__main__":
    print("gDel3D GPU-accelerated Delaunay Triangulation Benchmark")
    print(f"Version: {gdel3d.__version__}")
    print()

    # Initialize GPU
    print("Initializing GPU...")
    print(gdel3d.initialize_gpu())

    # Benchmark different sizes
    sizes = [100, 1_000, 10_000, 100_000]

    # Add 1M and 2M if user wants (commented by default)
    # sizes.extend([1_000_000, 2_000_000])

    results = []
    for size in sizes:
        try:
            avg_time, avg_tets = benchmark(size, num_runs=3)
            results.append((size, avg_time, avg_tets))
        except Exception as e:
            print(f"\nError with {size:,} points: {e}")

    # Summary table
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"{'Points':>12} | {'Time (ms)':>10} | {'Tets':>12} | {'Points/sec':>12}")
    print(f"{'-'*12}-+-{'-'*10}-+-{'-'*12}-+-{'-'*12}")

    for size, avg_time, avg_tets in results:
        throughput = size / avg_time
        print(f"{size:>12,} | {avg_time*1000:>10.1f} | {avg_tets:>12,.0f} | {throughput:>12,.0f}")

    print()
    print("Benchmark complete!")
