//! Benchmark program for gdel3d_wgpu performance testing.
//!
//! Run with: cargo run --release --example benchmark
//!
//! Options:
//!   cargo run --release --example benchmark           # Default: 100, 1k, 10k, 100k, 200k
//!   cargo run --release --example benchmark -- 2000000 # Custom size (2M points)
//!   cargo run --release --example benchmark -- quick  # Quick test: 100, 1k, 10k

use gdel3d_wgpu::{delaunay_3d, required_limits, types::GDelConfig};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::time::Instant;

fn generate_uniform_points(n: usize, seed: u64) -> Vec<[f32; 3]> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| [rng.gen(), rng.gen(), rng.gen()])
        .collect()
}

fn benchmark_size(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    num_points: usize,
    num_runs: usize,
    config: &GDelConfig,
) {
    println!("\n{}", "=".repeat(70));
    println!("Benchmark: {:>10} points ({} runs)", format_number(num_points), num_runs);
    println!("{}", "=".repeat(70));

    let mut timings = Vec::new();
    let mut num_tets_list = Vec::new();
    let mut num_failed_list = Vec::new();

    for run in 0..num_runs {
        // Generate points with different seed for each run
        let seed = 12345 + run as u64;
        let points = generate_uniform_points(num_points, seed);

        // Run benchmark
        print!("  Run {:>2}/{}: ", run + 1, num_runs);
        std::io::Write::flush(&mut std::io::stdout()).unwrap();

        let start = Instant::now();
        let result = pollster::block_on(delaunay_3d(device, queue, &points, config));
        let elapsed = start.elapsed();

        let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
        timings.push(elapsed_ms);
        num_tets_list.push(result.tets.len());
        num_failed_list.push(result.failed_verts.len());

        println!(
            "{:>7.1} ms | {:>8} tets | {:>4} failed",
            elapsed_ms,
            format_number(result.tets.len()),
            result.failed_verts.len()
        );
    }

    // Calculate statistics
    let avg_time = timings.iter().sum::<f64>() / timings.len() as f64;
    let min_time = timings.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_time = timings.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let variance = timings.iter().map(|&t| (t - avg_time).powi(2)).sum::<f64>() / timings.len() as f64;
    let std_dev = variance.sqrt();

    let avg_tets = num_tets_list.iter().sum::<usize>() / num_tets_list.len();
    let total_failed = num_failed_list.iter().sum::<usize>();

    println!("\nStatistics:");
    println!("  Time:       {:>7.1} ± {:.1} ms", avg_time, std_dev);
    println!("  Range:      {:>7.1} - {:.1} ms", min_time, max_time);
    println!("  Throughput: {:>7} points/sec", format_number((num_points as f64 / (avg_time / 1000.0)) as usize));
    println!("  Avg tets:   {:>7}", format_number(avg_tets));
    println!("  Total fail: {:>7}", total_failed);
}

fn format_number(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{}k", n / 1_000)
    } else {
        n.to_string()
    }
}

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║         gDel3D GPU Delaunay Triangulation Benchmark              ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");

    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();
    let benchmark_mode = if args.len() > 1 {
        args[1].as_str()
    } else {
        "default"
    };

    let sizes = match benchmark_mode {
        "quick" => vec![100, 1_000, 10_000],
        "default" => vec![100, 1_000, 10_000, 100_000, 200_000],
        "full" => vec![100, 1_000, 10_000, 100_000, 200_000, 500_000, 1_000_000],
        "extreme" => vec![100, 1_000, 10_000, 100_000, 200_000, 500_000, 1_000_000, 2_000_000],
        custom => {
            if let Ok(n) = custom.parse::<usize>() {
                vec![n]
            } else {
                eprintln!("Error: Invalid argument '{}'. Use 'quick', 'default', 'full', 'extreme', or a number.", custom);
                std::process::exit(1);
            }
        }
    };

    // Initialize GPU
    println!("\nInitializing GPU...");
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .expect("Failed to find GPU adapter");

    let info = adapter.get_info();
    println!("GPU: {} ({:?})", info.name, info.backend);

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("benchmark_device"),
        required_features: wgpu::Features::SUBGROUP,
        required_limits: required_limits(),
        memory_hints: Default::default(),
        experimental_features: Default::default(),
        trace: Default::default(),
    }))
    .expect("Failed to create device");

    println!("Device created successfully\n");

    // Configuration
    let config = GDelConfig {
        insertion_rule: gdel3d_wgpu::types::InsertionRule::Circumcenter,
        enable_flipping: true,
        enable_sorting: false,
        enable_hilbert_sorting: false,
        enable_splaying: true,
        max_insert_iterations: 100,
        max_flip_iterations: 10,
    };

    println!("Configuration:");
    println!("  Insertion rule:   {:?}", config.insertion_rule);
    println!("  Flipping:         {}", config.enable_flipping);
    println!("  Splaying:         {}", config.enable_splaying);
    println!("  Max iterations:   {}", config.max_insert_iterations);

    // Run benchmarks
    let num_runs = if sizes.iter().any(|&s| s >= 500_000) { 3 } else { 5 };

    for &size in &sizes {
        benchmark_size(&device, &queue, size, num_runs, &config);
    }

    // Summary table
    println!("\n{}", "=".repeat(70));
    println!("Summary");
    println!("{}", "=".repeat(70));
    println!("{:<15} | {:>12} | {:>15}", "Points", "Avg Time", "Throughput");
    println!("{}", "-".repeat(15) + "-+-" + &"-".repeat(12) + "-+-" + &"-".repeat(15));

    for &size in &sizes {
        // Re-run one iteration for summary
        let points = generate_uniform_points(size, 12345);
        let start = Instant::now();
        let result = pollster::block_on(delaunay_3d(&device, &queue, &points, &config));
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        let throughput = size as f64 / (elapsed_ms / 1000.0);

        println!(
            "{:<15} | {:>10.1} ms | {:>12}/sec",
            format_number(size),
            elapsed_ms,
            format_number(throughput as usize)
        );

        if result.failed_verts.len() > 0 {
            println!("  ⚠️  {} vertices failed", result.failed_verts.len());
        }
    }

    println!("\n{}", "=".repeat(70));
    println!("Benchmark complete!");
    println!("{}\n", "=".repeat(70));
}
