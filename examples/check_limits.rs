use pollster;

fn main() {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    })).unwrap();

    let limits = adapter.limits();

    println!("GPU: {:?}", adapter.get_info());
    println!("\nBuffer Limits:");
    println!("  max_buffer_size:                    {} MB ({} bytes)",
             limits.max_buffer_size / (1024 * 1024), limits.max_buffer_size);
    println!("  max_storage_buffer_binding_size:    {} MB ({} bytes)",
             limits.max_storage_buffer_binding_size / (1024 * 1024), limits.max_storage_buffer_binding_size);
    println!("  max_uniform_buffer_binding_size:    {} MB ({} bytes)",
             limits.max_uniform_buffer_binding_size / (1024 * 1024), limits.max_uniform_buffer_binding_size);

    // Calculate max points based on limits
    // Each point needs 8 tets * 16 bytes = 128 bytes per point
    let bytes_per_point = 128u64;
    let max_points_binding = (limits.max_storage_buffer_binding_size as u64) / bytes_per_point;
    let max_points_buffer = (limits.max_buffer_size as u64) / bytes_per_point;

    println!("\nTheoretical Maximum Points:");
    println!("  Limited by binding size: {} million points", max_points_binding / 1_000_000);
    println!("  Limited by buffer size:  {} million points", max_points_buffer / 1_000_000);
    println!("  Actual limit:            {} million points",
             max_points_binding.min(max_points_buffer) / 1_000_000);
}
