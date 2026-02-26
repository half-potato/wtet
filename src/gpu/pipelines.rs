use super::buffers::GpuBuffers;

/// All compute pipelines and their bind groups.
pub struct Pipelines {
    // Pipeline: init (make_first_tetra)
    pub init_pipeline: wgpu::ComputePipeline,
    pub init_bind_group: wgpu::BindGroup,
    pub init_params: wgpu::Buffer,

    // Pipeline: point location
    pub locate_pipeline: wgpu::ComputePipeline,
    pub locate_bind_group: wgpu::BindGroup,
    pub locate_params: wgpu::Buffer,

    // Pipeline: vote for point
    pub vote_pipeline: wgpu::ComputePipeline,
    pub vote_bind_group: wgpu::BindGroup,
    pub vote_params: wgpu::Buffer,

    // Pipeline: pick winner
    pub pick_pipeline: wgpu::ComputePipeline,
    pub pick_bind_group: wgpu::BindGroup,
    pub pick_params: wgpu::Buffer,

    // Pipeline: split tetra
    pub split_pipeline: wgpu::ComputePipeline,
    pub split_bind_group: wgpu::BindGroup,
    pub split_params: wgpu::Buffer,

    // Pipeline: flip check
    pub flip_pipeline: wgpu::ComputePipeline,
    pub flip_bind_group: wgpu::BindGroup,
    /// Alternate bind group with flip_queue/flip_queue_next swapped.
    pub flip_bind_group_b: wgpu::BindGroup,
    pub flip_params: wgpu::Buffer,

    // Pipeline: reset votes
    pub reset_votes_pipeline: wgpu::ComputePipeline,
    pub reset_votes_bind_group: wgpu::BindGroup,
    pub reset_votes_params: wgpu::Buffer,

    // Pipeline: gather failed
    pub gather_pipeline: wgpu::ComputePipeline,
    pub gather_bind_group: wgpu::BindGroup,
    pub gather_params: wgpu::Buffer,
}

impl Pipelines {
    pub fn new(
        device: &wgpu::Device,
        bufs: &GpuBuffers,
        num_points: u32,
        max_tets: u32,
    ) -> Self {
        // --- Init pipeline ---
        let init_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("init.wgsl"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/init.wgsl").into(),
            ),
        });

        let init_params = GpuBuffers::create_params_buffer(device, [num_points, max_tets, 0, 0]);

        let init_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("init_bgl"),
            entries: &[
                storage_ro_entry(0),  // points
                storage_rw_entry(1),  // tets
                storage_rw_entry(2),  // tet_opp
                storage_rw_entry(3),  // tet_info
                storage_rw_entry(4),  // vert_tet
                storage_rw_entry(5),  // counters
                uniform_entry(6),     // params
            ],
        });

        let init_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("init_bg"),
            layout: &init_bgl,
            entries: &[
                buf_entry(0, &bufs.points),
                buf_entry(1, &bufs.tets),
                buf_entry(2, &bufs.tet_opp),
                buf_entry(3, &bufs.tet_info),
                buf_entry(4, &bufs.vert_tet),
                buf_entry(5, &bufs.counters),
                buf_entry(6, &init_params),
            ],
        });

        let init_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("init_pl"),
            bind_group_layouts: &[&init_bgl],
            push_constant_ranges: &[],
        });

        let init_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("init"),
            layout: Some(&init_pl),
            module: &init_shader,
            entry_point: Some("make_first_tetra"),
            compilation_options: Default::default(),
            cache: None,
        });

        // --- Point location pipeline ---
        let locate_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("point_location.wgsl"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/point_location.wgsl").into(),
            ),
        });

        let locate_params = GpuBuffers::create_params_buffer(device, [num_points, 512, 0, 0]);

        let locate_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("locate_bgl"),
            entries: &[
                storage_ro_entry(0),  // points
                storage_ro_entry(1),  // tets
                storage_ro_entry(2),  // tet_opp
                storage_ro_entry(3),  // tet_info
                storage_rw_entry(4),  // vert_tet
                storage_ro_entry(5),  // uninserted
                uniform_entry(6),     // params
            ],
        });

        let locate_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("locate_bg"),
            layout: &locate_bgl,
            entries: &[
                buf_entry(0, &bufs.points),
                buf_entry(1, &bufs.tets),
                buf_entry(2, &bufs.tet_opp),
                buf_entry(3, &bufs.tet_info),
                buf_entry(4, &bufs.vert_tet),
                buf_entry(5, &bufs.uninserted),
                buf_entry(6, &locate_params),
            ],
        });

        let locate_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("locate_pl"),
            bind_group_layouts: &[&locate_bgl],
            push_constant_ranges: &[],
        });

        let locate_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("locate"),
            layout: Some(&locate_pl),
            module: &locate_shader,
            entry_point: Some("locate_points"),
            compilation_options: Default::default(),
            cache: None,
        });

        // --- Vote pipeline ---
        let vote_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("vote.wgsl"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/vote.wgsl").into(),
            ),
        });

        let vote_params = GpuBuffers::create_params_buffer(device, [num_points, 0, 0, 0]);

        let vote_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vote_bgl"),
            entries: &[
                storage_ro_entry(0),  // points
                storage_ro_entry(1),  // tets
                storage_ro_entry(2),  // tet_info
                storage_rw_entry(3),  // tet_vote
                storage_ro_entry(4),  // vert_tet
                storage_ro_entry(5),  // uninserted
                uniform_entry(6),     // params
            ],
        });

        let vote_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vote_bg"),
            layout: &vote_bgl,
            entries: &[
                buf_entry(0, &bufs.points),
                buf_entry(1, &bufs.tets),
                buf_entry(2, &bufs.tet_info),
                buf_entry(3, &bufs.tet_vote),
                buf_entry(4, &bufs.vert_tet),
                buf_entry(5, &bufs.uninserted),
                buf_entry(6, &vote_params),
            ],
        });

        let vote_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("vote_pl"),
            bind_group_layouts: &[&vote_bgl],
            push_constant_ranges: &[],
        });

        let vote_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("vote"),
            layout: Some(&vote_pl),
            module: &vote_shader,
            entry_point: Some("vote_for_point"),
            compilation_options: Default::default(),
            cache: None,
        });

        // --- Pick winner pipeline ---
        let pick_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pick_winner.wgsl"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/pick_winner.wgsl").into(),
            ),
        });

        let pick_params = GpuBuffers::create_params_buffer(device, [max_tets, 0, 0, 0]);

        let pick_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pick_bgl"),
            entries: &[
                storage_ro_entry(0),  // tets
                storage_ro_entry(1),  // tet_info
                storage_ro_entry(2),  // tet_vote (read-only i32)
                storage_ro_entry(3),  // uninserted
                storage_rw_entry(4),  // insert_list
                storage_rw_entry(5),  // counters
                uniform_entry(6),     // params
            ],
        });

        let pick_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pick_bg"),
            layout: &pick_bgl,
            entries: &[
                buf_entry(0, &bufs.tets),
                buf_entry(1, &bufs.tet_info),
                buf_entry(2, &bufs.tet_vote),
                buf_entry(3, &bufs.uninserted),
                buf_entry(4, &bufs.insert_list),
                buf_entry(5, &bufs.counters),
                buf_entry(6, &pick_params),
            ],
        });

        let pick_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pick_pl"),
            bind_group_layouts: &[&pick_bgl],
            push_constant_ranges: &[],
        });

        let pick_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pick_winner"),
            layout: Some(&pick_pl),
            module: &pick_shader,
            entry_point: Some("pick_winner_point"),
            compilation_options: Default::default(),
            cache: None,
        });

        // --- Split pipeline ---
        let split_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("split.wgsl"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/split.wgsl").into(),
            ),
        });

        let split_params = GpuBuffers::create_params_buffer(device, [0, 0, 0, 0]);

        let split_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("split_bgl"),
            entries: &[
                storage_ro_entry(0),  // points
                storage_rw_entry(1),  // tets
                storage_rw_entry(2),  // tet_opp
                storage_rw_entry(3),  // tet_info
                storage_rw_entry(4),  // vert_tet
                storage_ro_entry(5),  // insert_list
                storage_rw_entry(6),  // free_stack
                storage_rw_entry(7),  // counters
                storage_rw_entry(8),  // flip_queue
                uniform_entry(9),     // params
            ],
        });

        let split_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("split_bg"),
            layout: &split_bgl,
            entries: &[
                buf_entry(0, &bufs.points),
                buf_entry(1, &bufs.tets),
                buf_entry(2, &bufs.tet_opp),
                buf_entry(3, &bufs.tet_info),
                buf_entry(4, &bufs.vert_tet),
                buf_entry(5, &bufs.insert_list),
                buf_entry(6, &bufs.free_stack),
                buf_entry(7, &bufs.counters),
                buf_entry(8, &bufs.flip_queue),
                buf_entry(9, &split_params),
            ],
        });

        let split_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("split_pl"),
            bind_group_layouts: &[&split_bgl],
            push_constant_ranges: &[],
        });

        let split_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("split"),
            layout: Some(&split_pl),
            module: &split_shader,
            entry_point: Some("split_tetra"),
            compilation_options: Default::default(),
            cache: None,
        });

        // --- Flip pipeline ---
        let flip_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("flip.wgsl"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/flip.wgsl").into(),
            ),
        });

        let flip_params = GpuBuffers::create_params_buffer(device, [0, 0, 0, 0]);

        let flip_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("flip_bgl"),
            entries: &[
                storage_ro_entry(0),  // points
                storage_rw_entry(1),  // tets
                storage_rw_entry(2),  // tet_opp
                storage_rw_entry(3),  // tet_info
                storage_rw_entry(4),  // free_stack
                storage_rw_entry(5),  // counters
                storage_ro_entry(6),  // flip_queue
                storage_rw_entry(7),  // flip_queue_next
                storage_rw_entry(8),  // flip_count
                uniform_entry(9),     // params
            ],
        });

        let flip_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("flip_bg"),
            layout: &flip_bgl,
            entries: &[
                buf_entry(0, &bufs.points),
                buf_entry(1, &bufs.tets),
                buf_entry(2, &bufs.tet_opp),
                buf_entry(3, &bufs.tet_info),
                buf_entry(4, &bufs.free_stack),
                buf_entry(5, &bufs.counters),
                buf_entry(6, &bufs.flip_queue),
                buf_entry(7, &bufs.flip_queue_next),
                buf_entry(8, &bufs.flip_count),
                buf_entry(9, &flip_params),
            ],
        });

        // Alternate bind group: flip_queue and flip_queue_next swapped
        let flip_bind_group_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("flip_bg_b"),
            layout: &flip_bgl,
            entries: &[
                buf_entry(0, &bufs.points),
                buf_entry(1, &bufs.tets),
                buf_entry(2, &bufs.tet_opp),
                buf_entry(3, &bufs.tet_info),
                buf_entry(4, &bufs.free_stack),
                buf_entry(5, &bufs.counters),
                buf_entry(6, &bufs.flip_queue_next),  // swapped: read from next
                buf_entry(7, &bufs.flip_queue),        // swapped: write to original
                buf_entry(8, &bufs.flip_count),
                buf_entry(9, &flip_params),
            ],
        });

        let flip_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("flip_pl"),
            bind_group_layouts: &[&flip_bgl],
            push_constant_ranges: &[],
        });

        let flip_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("flip"),
            layout: Some(&flip_pl),
            module: &flip_shader,
            entry_point: Some("flip_check"),
            compilation_options: Default::default(),
            cache: None,
        });

        // --- Reset votes pipeline ---
        let reset_votes_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("reset_votes.wgsl"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/reset_votes.wgsl").into(),
            ),
        });

        let reset_votes_params = GpuBuffers::create_params_buffer(device, [max_tets, 0, 0, 0]);

        let reset_votes_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("reset_votes_bgl"),
            entries: &[
                storage_rw_entry(0),  // tet_vote
                uniform_entry(1),     // params
            ],
        });

        let reset_votes_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("reset_votes_bg"),
            layout: &reset_votes_bgl,
            entries: &[
                buf_entry(0, &bufs.tet_vote),
                buf_entry(1, &reset_votes_params),
            ],
        });

        let reset_votes_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("reset_votes_pl"),
            bind_group_layouts: &[&reset_votes_bgl],
            push_constant_ranges: &[],
        });

        let reset_votes_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("reset_votes"),
                layout: Some(&reset_votes_pl),
                module: &reset_votes_shader,
                entry_point: Some("reset_votes"),
                compilation_options: Default::default(),
                cache: None,
            });

        // --- Gather failed pipeline ---
        let gather_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gather.wgsl"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/gather.wgsl").into(),
            ),
        });

        let gather_params =
            GpuBuffers::create_params_buffer(device, [max_tets, num_points, 0, 0]);

        let gather_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gather_bgl"),
            entries: &[
                storage_ro_entry(0),  // points
                storage_ro_entry(1),  // tets
                storage_ro_entry(2),  // tet_opp
                storage_ro_entry(3),  // tet_info
                storage_rw_entry(4),  // failed_verts
                storage_rw_entry(5),  // counters
                uniform_entry(6),     // params
            ],
        });

        let gather_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gather_bg"),
            layout: &gather_bgl,
            entries: &[
                buf_entry(0, &bufs.points),
                buf_entry(1, &bufs.tets),
                buf_entry(2, &bufs.tet_opp),
                buf_entry(3, &bufs.tet_info),
                buf_entry(4, &bufs.failed_verts),
                buf_entry(5, &bufs.counters),
                buf_entry(6, &gather_params),
            ],
        });

        let gather_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gather_pl"),
            bind_group_layouts: &[&gather_bgl],
            push_constant_ranges: &[],
        });

        let gather_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gather"),
            layout: Some(&gather_pl),
            module: &gather_shader,
            entry_point: Some("gather_failed"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            init_pipeline,
            init_bind_group,
            init_params,
            locate_pipeline,
            locate_bind_group,
            locate_params,
            vote_pipeline,
            vote_bind_group,
            vote_params,
            pick_pipeline,
            pick_bind_group,
            pick_params,
            split_pipeline,
            split_bind_group,
            split_params,
            flip_pipeline,
            flip_bind_group,
            flip_bind_group_b,
            flip_params,
            reset_votes_pipeline,
            reset_votes_bind_group,
            reset_votes_params,
            gather_pipeline,
            gather_bind_group,
            gather_params,
        }
    }
}

// --- Helper functions for bind group layout entries ---

fn storage_ro_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_rw_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn buf_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}
