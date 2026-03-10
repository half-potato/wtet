use std::collections::HashMap;
use std::time::{Duration, Instant};

/// GPU profiler using timestamp queries
pub struct GpuProfiler {
    query_set: wgpu::QuerySet,
    resolve_buffer: wgpu::Buffer,
    readback_buffer: wgpu::Buffer,
    query_capacity: u32,
    next_query_idx: u32,
    pending_queries: Vec<(String, u32, u32)>, // (label, start_idx, end_idx)
    accumulated_times: HashMap<String, Vec<f64>>, // label -> times in ms
}

impl GpuProfiler {
    pub fn new(device: &wgpu::Device, capacity: u32) -> Self {
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("profiler_queries"),
            ty: wgpu::QueryType::Timestamp,
            count: capacity,
        });

        let resolve_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("profiler_resolve"),
            size: (capacity as u64) * 8, // 8 bytes per timestamp
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("profiler_readback"),
            size: (capacity as u64) * 8,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Self {
            query_set,
            resolve_buffer,
            readback_buffer,
            query_capacity: capacity,
            next_query_idx: 0,
            pending_queries: Vec::new(),
            accumulated_times: HashMap::new(),
        }
    }

    /// Begin a GPU profiling scope. Returns (start_idx, end_idx) for timestamp writes.
    pub fn begin_scope(&mut self, label: String) -> Option<(u32, u32)> {
        if self.next_query_idx + 2 > self.query_capacity {
            eprintln!("[PROFILER] Query capacity exceeded, skipping '{}'", label);
            return None;
        }

        let start_idx = self.next_query_idx;
        let end_idx = self.next_query_idx + 1;
        self.next_query_idx += 2;

        self.pending_queries.push((label, start_idx, end_idx));
        Some((start_idx, end_idx))
    }

    /// Resolve and readback all pending queries
    pub async fn collect(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, timestamp_period: f32) {
        if self.pending_queries.is_empty() {
            return;
        }

        // Resolve queries
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("profiler_resolve"),
        });

        encoder.resolve_query_set(
            &self.query_set,
            0..self.next_query_idx,
            &self.resolve_buffer,
            0,
        );

        encoder.copy_buffer_to_buffer(
            &self.resolve_buffer,
            0,
            &self.readback_buffer,
            0,
            (self.next_query_idx as u64) * 8,
        );

        queue.submit(Some(encoder.finish()));

        // Readback
        let buffer_slice = self.readback_buffer.slice(..);
        let (sender, receiver) = futures_channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
        device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        receiver.await.unwrap().unwrap();

        let data = buffer_slice.get_mapped_range();
        let timestamps: &[u64] = bytemuck::cast_slice(&data);

        // Calculate durations
        for (label, start_idx, end_idx) in &self.pending_queries {
            let start = timestamps[*start_idx as usize];
            let end = timestamps[*end_idx as usize];
            let duration_ns = (end - start) as f64 * timestamp_period as f64;
            let duration_ms = duration_ns / 1_000_000.0;

            self.accumulated_times
                .entry(label.clone())
                .or_insert_with(Vec::new)
                .push(duration_ms);
        }

        drop(data);
        self.readback_buffer.unmap();

        // Reset for next frame
        self.next_query_idx = 0;
        self.pending_queries.clear();
    }

    /// Print accumulated statistics
    pub fn print_summary(&self) {
        if self.accumulated_times.is_empty() {
            return;
        }

        eprintln!("\n========== GPU PROFILING SUMMARY ==========");

        let mut entries: Vec<_> = self.accumulated_times.iter().collect();
        entries.sort_by(|a, b| {
            let sum_a: f64 = a.1.iter().sum();
            let sum_b: f64 = b.1.iter().sum();
            sum_b.partial_cmp(&sum_a).unwrap()
        });

        for (label, times) in entries {
            let count = times.len();
            let total: f64 = times.iter().sum();
            let avg = total / count as f64;
            let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

            eprintln!(
                "{:40} | calls: {:5} | total: {:8.2} ms | avg: {:6.3} ms | min: {:6.3} ms | max: {:6.3} ms",
                label, count, total, avg, min, max
            );
        }

        let grand_total: f64 = self.accumulated_times.values().map(|v| v.iter().sum::<f64>()).sum();
        eprintln!("{:=<120}", "");
        eprintln!("{:40} | {:59.2} ms", "TOTAL GPU TIME", grand_total);
        eprintln!("{:=<120}\n", "");
    }
}

/// CPU-side profiler using std::time::Instant
pub struct CpuProfiler {
    accumulated_times: HashMap<String, Vec<Duration>>,
    current_scopes: HashMap<String, Instant>,
}

impl CpuProfiler {
    pub fn new() -> Self {
        Self {
            accumulated_times: HashMap::new(),
            current_scopes: HashMap::new(),
        }
    }

    pub fn begin(&mut self, label: &str) {
        self.current_scopes.insert(label.to_string(), Instant::now());
    }

    pub fn end(&mut self, label: &str) {
        if let Some(start) = self.current_scopes.remove(label) {
            let duration = start.elapsed();
            self.accumulated_times
                .entry(label.to_string())
                .or_insert_with(Vec::new)
                .push(duration);
        }
    }

    pub fn print_summary(&self) {
        if self.accumulated_times.is_empty() {
            return;
        }

        eprintln!("\n========== CPU PROFILING SUMMARY ==========");

        let mut entries: Vec<_> = self.accumulated_times.iter().collect();
        entries.sort_by(|a, b| {
            let sum_a: Duration = a.1.iter().sum();
            let sum_b: Duration = b.1.iter().sum();
            sum_b.cmp(&sum_a)
        });

        for (label, times) in entries {
            let count = times.len();
            let total: Duration = times.iter().sum();
            let avg = total / count as u32;
            let min = times.iter().min().unwrap();
            let max = times.iter().max().unwrap();

            eprintln!(
                "{:40} | calls: {:5} | total: {:8.2} ms | avg: {:6.3} ms | min: {:6.3} ms | max: {:6.3} ms",
                label,
                count,
                total.as_secs_f64() * 1000.0,
                avg.as_secs_f64() * 1000.0,
                min.as_secs_f64() * 1000.0,
                max.as_secs_f64() * 1000.0
            );
        }

        let grand_total: Duration = self.accumulated_times.values().map(|v| v.iter().copied().sum::<Duration>()).sum();
        eprintln!("{:=<120}", "");
        eprintln!("{:40} | {:59.2} ms", "TOTAL CPU TIME", grand_total.as_secs_f64() * 1000.0);
        eprintln!("{:=<120}\n", "");
    }
}
