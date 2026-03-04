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

    // Pipeline: update vert free list (allocates tet blocks for inserting vertices)
    pub update_vert_free_pipeline: wgpu::ComputePipeline,
    pub update_vert_free_bind_group: wgpu::BindGroup,
    pub update_vert_free_params: wgpu::Buffer,

    // Pipeline: mark split (marks tets being split for concurrent detection)
    pub mark_split_pipeline: wgpu::ComputePipeline,
    pub mark_split_bind_group: wgpu::BindGroup,
    pub mark_split_params: wgpu::Buffer,

    // Pipeline: split points (updates vert_tet for uninserted vertices whose tets are splitting)
    pub split_points_pipeline: wgpu::ComputePipeline,
    pub split_points_bind_group: wgpu::BindGroup,

    // Pipeline: split tetra
    pub split_pipeline: wgpu::ComputePipeline,
    pub split_bind_group: wgpu::BindGroup,
    pub split_params: wgpu::Buffer,

    // Pipeline: update uninserted vert_tet (fixes dead tet references)
    pub update_uninserted_vert_tet_pipeline: wgpu::ComputePipeline,
    pub update_uninserted_vert_tet_bind_group: wgpu::BindGroup,
    pub update_uninserted_vert_tet_params: wgpu::Buffer,

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

    // Pipeline: check delaunay fast (flip voting)
    pub check_delaunay_fast_pipeline: wgpu::ComputePipeline,
    pub check_delaunay_fast_bind_group: wgpu::BindGroup,
    pub check_delaunay_fast_params: wgpu::Buffer,

    // Pipeline: check delaunay exact (flip voting with DD + SoS)
    pub check_delaunay_exact_pipeline: wgpu::ComputePipeline,
    pub check_delaunay_exact_bind_group: wgpu::BindGroup,
    pub check_delaunay_exact_params: wgpu::Buffer,

    // Pipeline: allocate flip23 slot
    pub allocate_flip23_slot_pipeline: wgpu::ComputePipeline,
    pub allocate_flip23_slot_bind_group: wgpu::BindGroup,
    pub allocate_flip23_slot_params: wgpu::Buffer,

    // Pipeline: compact if negative
    pub compact_if_negative_pipeline: wgpu::ComputePipeline,
    pub compact_if_negative_bind_group: wgpu::BindGroup,
    pub compact_if_negative_params: wgpu::Buffer,

    // Pipeline: relocate points fast
    pub relocate_points_fast_pipeline: wgpu::ComputePipeline,
    pub relocate_points_fast_bind_group: wgpu::BindGroup,
    pub relocate_points_fast_params: wgpu::Buffer,

    // Pipeline: update opp (fix adjacency after flips)
    pub update_opp_pipeline: wgpu::ComputePipeline,
    pub update_opp_bind_group: wgpu::BindGroup,
    pub update_opp_params: wgpu::Buffer,

    // Pipeline: update flip trace (build flip history chains for relocateAll)
    pub update_flip_trace_pipeline: wgpu::ComputePipeline,
    pub update_flip_trace_bind_group: wgpu::BindGroup,
    pub update_flip_trace_params: wgpu::Buffer,

    // Pipeline: collect free slots (compactTetras step 1)
    pub collect_free_slots_pipeline: wgpu::ComputePipeline,
    pub collect_free_slots_bind_group: wgpu::BindGroup,
    pub collect_free_slots_params: wgpu::Buffer,

    // Pipeline: make compact map (compactTetras step 2)
    pub make_compact_map_pipeline: wgpu::ComputePipeline,
    pub make_compact_map_bind_group: wgpu::BindGroup,
    pub make_compact_map_params: wgpu::Buffer,

    // Pipeline: compact tets (compactTetras step 3)
    pub compact_tets_pipeline: wgpu::ComputePipeline,
    pub compact_tets_bind_group: wgpu::BindGroup,
    pub compact_tets_params: wgpu::Buffer,

    // Pipeline: mark special tets (between fast/exact flipping)
    pub mark_special_tets_pipeline: wgpu::ComputePipeline,
    pub mark_special_tets_bind_group: wgpu::BindGroup,
    pub mark_special_tets_params: wgpu::Buffer,

    // Pipeline: mark rejected flips
    pub mark_rejected_flips_pipeline: wgpu::ComputePipeline,
    pub mark_rejected_flips_bind_group: wgpu::BindGroup,
    pub mark_rejected_flips_params: wgpu::Buffer,
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
                storage_rw_entry(5),  // free_arr (unused)
                storage_rw_entry(6),  // vert_free_arr (unused)
                storage_rw_entry(7),  // counters
                uniform_entry(8),     // params
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
                buf_entry(5, &bufs.free_arr),
                buf_entry(6, &bufs.vert_free_arr),
                buf_entry(7, &bufs.counters),
                buf_entry(8, &init_params),
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
                storage_ro_entry(6),  // vert_tet (ADDED for vertex-parallel iteration)
                uniform_entry(7),     // params
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
                buf_entry(6, &bufs.vert_tet),
                buf_entry(7, &pick_params),
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

        // --- Update vert free list pipeline ---
        let update_vert_free_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("update_vert_free_list.wgsl"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/update_vert_free_list.wgsl").into(),
            ),
        });

        let update_vert_free_params = GpuBuffers::create_params_buffer(device, [0, 0, 0, 0]);

        let update_vert_free_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("update_vert_free_bgl"),
            entries: &[
                storage_ro_entry(0),  // insert_list
                storage_rw_entry(1),  // vert_free_arr
                storage_rw_entry(2),  // free_arr
                uniform_entry(3),     // params (x = num_insertions, y = start_free_idx)
            ],
        });

        let update_vert_free_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("update_vert_free_bg"),
            layout: &update_vert_free_bgl,
            entries: &[
                buf_entry(0, &bufs.insert_list),
                buf_entry(1, &bufs.vert_free_arr),
                buf_entry(2, &bufs.free_arr),
                buf_entry(3, &update_vert_free_params),
            ],
        });

        let update_vert_free_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("update_vert_free_pl"),
            bind_group_layouts: &[&update_vert_free_bgl],
            push_constant_ranges: &[],
        });

        let update_vert_free_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("update_vert_free_list"),
            layout: Some(&update_vert_free_pl),
            module: &update_vert_free_shader,
            entry_point: Some("update_vert_free_list"),
            compilation_options: Default::default(),
            cache: None,
        });

        // --- Mark split pipeline ---
        let mark_split_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mark_split.wgsl"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/mark_split.wgsl").into(),
            ),
        });

        let mark_split_params = GpuBuffers::create_params_buffer(device, [0, 0, 0, 0]);

        let mark_split_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mark_split_bgl"),
            entries: &[
                storage_ro_entry(0),  // insert_list
                storage_rw_entry(1),  // tet_to_vert
                uniform_entry(2),     // params
            ],
        });

        let mark_split_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mark_split_bg"),
            layout: &mark_split_bgl,
            entries: &[
                buf_entry(0, &bufs.insert_list),
                buf_entry(1, &bufs.tet_to_vert),
                buf_entry(2, &mark_split_params),
            ],
        });

        let mark_split_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mark_split_pl"),
            bind_group_layouts: &[&mark_split_bgl],
            push_constant_ranges: &[],
        });

        let mark_split_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("mark_split"),
            layout: Some(&mark_split_pl),
            module: &mark_split_shader,
            entry_point: Some("mark_split"),
            compilation_options: Default::default(),
            cache: None,
        });

        // --- Split points pipeline ---
        let split_points_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("split_points.wgsl"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/split_points.wgsl").into(),
            ),
        });

        let split_points_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("split_points_bgl"),
            entries: &[
                storage_ro_entry(0),  // points
                storage_ro_entry(1),  // tets
                storage_ro_entry(2),  // tet_opp
                storage_ro_entry(3),  // tet_info
                storage_rw_entry(4),  // vert_tet
                storage_ro_entry(5),  // free_arr
                storage_ro_entry(6),  // vert_free_arr
                storage_rw_entry(7),  // counters
                storage_ro_entry(8),  // uninserted
                storage_ro_entry(9),  // tet_to_vert
            ],
        });

        let split_points_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("split_points_bg"),
            layout: &split_points_bgl,
            entries: &[
                buf_entry(0, &bufs.points),
                buf_entry(1, &bufs.tets),
                buf_entry(2, &bufs.tet_opp),
                buf_entry(3, &bufs.tet_info),
                buf_entry(4, &bufs.vert_tet),
                buf_entry(5, &bufs.free_arr),
                buf_entry(6, &bufs.vert_free_arr),
                buf_entry(7, &bufs.counters),
                buf_entry(8, &bufs.uninserted),
                buf_entry(9, &bufs.tet_to_vert),
            ],
        });

        let split_points_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("split_points_pl"),
            bind_group_layouts: &[&split_points_bgl],
            push_constant_ranges: &[],
        });

        let split_points_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("split_points"),
            layout: Some(&split_points_pl),
            module: &split_points_shader,
            entry_point: Some("main"),
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
                storage_rw_entry(0),  // tets
                storage_rw_entry(1),  // tet_opp (flat atomic)
                storage_rw_entry(2),  // tet_info
                storage_rw_entry(3),  // vert_tet
                storage_ro_entry(4),  // insert_list
                storage_rw_entry(5),  // free_arr
                storage_rw_entry(6),  // vert_free_arr
                storage_rw_entry(7),  // counters
                storage_rw_entry(8),  // flip_queue
                storage_rw_entry(9),  // tet_to_vert (read for neighbor check, write to clear)
                uniform_entry(10),    // params
                // NOTE: Removed breadcrumbs and thread_debug to stay under 10 storage buffer limit
            ],
        });

        let split_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("split_bg"),
            layout: &split_bgl,
            entries: &[
                buf_entry(0, &bufs.tets),
                buf_entry(1, &bufs.tet_opp),
                buf_entry(2, &bufs.tet_info),
                buf_entry(3, &bufs.vert_tet),
                buf_entry(4, &bufs.insert_list),
                buf_entry(5, &bufs.free_arr),
                buf_entry(6, &bufs.vert_free_arr),
                buf_entry(7, &bufs.counters),
                buf_entry(8, &bufs.flip_queue),
                buf_entry(9, &bufs.tet_to_vert),
                buf_entry(10, &split_params),
                // NOTE: Removed breadcrumbs and thread_debug to stay under 10 storage buffer limit
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

        // --- Update uninserted vert_tet pipeline ---
        let update_uninserted_vert_tet_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("update_uninserted_vert_tet.wgsl"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/update_uninserted_vert_tet.wgsl").into(),
            ),
        });

        let update_uninserted_vert_tet_params = GpuBuffers::create_params_buffer(device, [0, 0, 0, 0]);

        let update_uninserted_vert_tet_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("update_uninserted_vert_tet_bgl"),
            entries: &[
                storage_ro_entry(0),  // tet_info
                storage_ro_entry(1),  // uninserted
                storage_rw_entry(2),  // vert_tet
                uniform_entry(3),     // params
            ],
        });

        let update_uninserted_vert_tet_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("update_uninserted_vert_tet_bg"),
            layout: &update_uninserted_vert_tet_bgl,
            entries: &[
                buf_entry(0, &bufs.tet_info),
                buf_entry(1, &bufs.uninserted),
                buf_entry(2, &bufs.vert_tet),
                buf_entry(3, &update_uninserted_vert_tet_params),
            ],
        });

        let update_uninserted_vert_tet_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("update_uninserted_vert_tet_pl"),
            bind_group_layouts: &[&update_uninserted_vert_tet_bgl],
            push_constant_ranges: &[],
        });

        let update_uninserted_vert_tet_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("update_uninserted_vert_tet"),
            layout: Some(&update_uninserted_vert_tet_pl),
            module: &update_uninserted_vert_tet_shader,
            entry_point: Some("update_uninserted_vert_tet"),
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
                storage_rw_entry(4),  // free_arr
                storage_rw_entry(5),  // vert_free_arr
                storage_rw_entry(6),  // counters
                storage_ro_entry(7),  // flip_queue
                storage_rw_entry(8),  // flip_queue_next
                storage_rw_entry(9),  // flip_count
                uniform_entry(10),    // params
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
                buf_entry(4, &bufs.free_arr),
                buf_entry(5, &bufs.vert_free_arr),
                buf_entry(6, &bufs.counters),
                buf_entry(7, &bufs.flip_queue),
                buf_entry(8, &bufs.flip_queue_next),
                buf_entry(9, &bufs.flip_count),
                buf_entry(10, &flip_params),
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
                buf_entry(4, &bufs.free_arr),
                buf_entry(5, &bufs.vert_free_arr),
                buf_entry(6, &bufs.counters),
                buf_entry(7, &bufs.flip_queue_next),  // swapped: read from next
                buf_entry(8, &bufs.flip_queue),        // swapped: write to original
                buf_entry(9, &bufs.flip_count),
                buf_entry(10, &flip_params),
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


        // Get shader compilation info first
        let shader_info = pollster::block_on(gather_shader.get_compilation_info());
        if !shader_info.messages.is_empty() {
            eprintln!("    Gather shader compilation messages:");
            for msg in &shader_info.messages {
                eprintln!("      [{:?}] {}", msg.message_type, msg.message);
            }
        }

        let gather_pipeline = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("gather"),
                layout: Some(&gather_pl),
                module: &gather_shader,
                entry_point: Some("gather_failed"),
                compilation_options: Default::default(),
                cache: None,
            })
        })) {
            Ok(pipeline) => {
                pipeline
            }
            Err(e) => {
                eprintln!("    ✗ PANIC during gather pipeline creation:");
                eprintln!("      {:?}", e);
                panic!("Failed to create gather pipeline");
            }
        };

        // ==================== NEW PIPELINES (14 KERNELS) - PROPER BIND GROUPS ====================


        // --- 1. Collect free slots (compaction) ---
        let collect_free_slots_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("collect_free_slots.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/collect_free_slots.wgsl").into()),
        });
        let collect_free_slots_params = GpuBuffers::create_params_buffer(device, [0, 0, 0, 0]);
        let collect_free_slots_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("collect_free_slots_bgl"),
            entries: &[storage_ro_entry(0), storage_ro_entry(1), storage_rw_entry(2), uniform_entry(3)],
        });
        let collect_free_slots_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("collect_free_slots_bg"),
            layout: &collect_free_slots_bgl,
            entries: &[
                buf_entry(0, &bufs.tet_info),
                buf_entry(1, &bufs.prefix_sum_data),
                buf_entry(2, &bufs.free_arr),
                buf_entry(3, &collect_free_slots_params),
            ],
        });
        let collect_free_slots_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("collect_free_slots_pl"),
            bind_group_layouts: &[&collect_free_slots_bgl],
            push_constant_ranges: &[],
        });
        let collect_free_slots_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("collect_free_slots"),
            layout: Some(&collect_free_slots_pl),
            module: &collect_free_slots_shader,
            entry_point: Some("collect_free_slots"),
            compilation_options: Default::default(),
            cache: None,
        });

        // --- 2. Make compact map (compaction) ---
        let make_compact_map_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("make_compact_map.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/make_compact_map.wgsl").into()),
        });
        let make_compact_map_params = GpuBuffers::create_params_buffer(device, [0, 0, 0, 0]);
        let make_compact_map_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("make_compact_map_bgl"),
            entries: &[storage_ro_entry(0), storage_rw_entry(1), storage_ro_entry(2), uniform_entry(3)],
        });
        let make_compact_map_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("make_compact_map_bg"),
            layout: &make_compact_map_bgl,
            entries: &[
                buf_entry(0, &bufs.tet_info),
                buf_entry(1, &bufs.prefix_sum_data),
                buf_entry(2, &bufs.free_arr),
                buf_entry(3, &make_compact_map_params),
            ],
        });
        let make_compact_map_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("make_compact_map_pl"),
            bind_group_layouts: &[&make_compact_map_bgl],
            push_constant_ranges: &[],
        });
        let make_compact_map_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("make_compact_map"),
            layout: Some(&make_compact_map_pl),
            module: &make_compact_map_shader,
            entry_point: Some("make_compact_map"),
            compilation_options: Default::default(),
            cache: None,
        });

        // --- 3. Compact tets (compaction) ---
        let compact_tets_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("compact_tets.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/compact_tets.wgsl").into()),
        });
        let compact_tets_params = GpuBuffers::create_params_buffer(device, [0, 0, 0, 0]);
        let compact_tets_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("compact_tets_bgl"),
            entries: &[storage_ro_entry(0), storage_ro_entry(1), storage_rw_entry(2), storage_rw_entry(3), uniform_entry(4)],
        });
        let compact_tets_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("compact_tets_bg"),
            layout: &compact_tets_bgl,
            entries: &[
                buf_entry(0, &bufs.tet_info),
                buf_entry(1, &bufs.prefix_sum_data),
                buf_entry(2, &bufs.tets),
                buf_entry(3, &bufs.tet_opp),
                buf_entry(4, &compact_tets_params),
            ],
        });
        let compact_tets_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("compact_tets_pl"),
            bind_group_layouts: &[&compact_tets_bgl],
            push_constant_ranges: &[],
        });
        let compact_tets_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("compact_tets"),
            layout: Some(&compact_tets_pl),
            module: &compact_tets_shader,
            entry_point: Some("compact_tets"),
            compilation_options: Default::default(),
            cache: None,
        });

        // --- 4. Mark special tets (flip management) ---
        let mark_special_tets_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mark_special_tets.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/mark_special_tets.wgsl").into()),
        });
        let mark_special_tets_params = GpuBuffers::create_params_buffer(device, [max_tets, 0, 0, 0]);
        let mark_special_tets_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mark_special_tets_bgl"),
            entries: &[
                storage_rw_entry(0),  // tet_info
                storage_rw_entry(1),  // tet_opp
                storage_rw_entry(2),  // act_tet_vec
                storage_rw_entry(3),  // counters
                uniform_entry(4),     // params
            ],
        });
        let mark_special_tets_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mark_special_tets_bg"),
            layout: &mark_special_tets_bgl,
            entries: &[
                buf_entry(0, &bufs.tet_info),
                buf_entry(1, &bufs.tet_opp),
                buf_entry(2, &bufs.act_tet_vec),
                buf_entry(3, &bufs.counters),
                buf_entry(4, &mark_special_tets_params),
            ],
        });
        let mark_special_tets_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mark_special_tets_pl"),
            bind_group_layouts: &[&mark_special_tets_bgl],
            push_constant_ranges: &[],
        });
        let mark_special_tets_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("mark_special_tets"),
            layout: Some(&mark_special_tets_pl),
            module: &mark_special_tets_shader,
            entry_point: Some("mark_special_tets"),
            compilation_options: Default::default(),
            cache: None,
        });

        // --- 5. Update flip trace (flip management) ---
        let update_flip_trace_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("update_flip_trace.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/update_flip_trace.wgsl").into()),
        });
        let update_flip_trace_params = GpuBuffers::create_params_buffer(device, [0, 0, 0, 0]);
        let update_flip_trace_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("update_flip_trace_bgl"),
            entries: &[storage_rw_entry(0), storage_rw_entry(1), uniform_entry(2)],
        });
        let update_flip_trace_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("update_flip_trace_bg"),
            layout: &update_flip_trace_bgl,
            entries: &[
                buf_entry(0, &bufs.flip_arr),
                buf_entry(1, &bufs.tet_to_flip),
                buf_entry(2, &update_flip_trace_params),
            ],
        });
        let update_flip_trace_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("update_flip_trace_pl"),
            bind_group_layouts: &[&update_flip_trace_bgl],
            push_constant_ranges: &[],
        });
        let update_flip_trace_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("update_flip_trace"),
            layout: Some(&update_flip_trace_pl),
            module: &update_flip_trace_shader,
            entry_point: Some("update_flip_trace"),
            compilation_options: Default::default(),
            cache: None,
        });

        // --- 12. Update opp (CRITICAL - flip adjacency) ---
        let update_opp_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("update_opp.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/update_opp.wgsl").into()),
        });
        let update_opp_params = GpuBuffers::create_params_buffer(device, [0, 0, 0, 0]);
        let update_opp_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("update_opp_bgl"),
            entries: &[
                storage_ro_entry(0), // flip_vec (read-only)
                storage_rw_entry(1), // opp_arr (read-write)
                storage_ro_entry(2), // tet_msg_arr
                storage_ro_entry(3), // encoded_face_vi_arr
                uniform_entry(4),    // params
            ],
        });
        let update_opp_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("update_opp_bg"),
            layout: &update_opp_bgl,
            entries: &[
                buf_entry(0, &bufs.flip_arr),
                buf_entry(1, &bufs.tet_opp),
                buf_entry(2, &bufs.tet_msg_arr),
                buf_entry(3, &bufs.encoded_face_vi_arr),
                buf_entry(4, &update_opp_params),
            ],
        });
        let update_opp_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("update_opp_pl"),
            bind_group_layouts: &[&update_opp_bgl],
            push_constant_ranges: &[],
        });
        let update_opp_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("update_opp"),
            layout: Some(&update_opp_pl),
            module: &update_opp_shader,
            entry_point: Some("update_opp"),
            compilation_options: Default::default(),
            cache: None,
        });

        // --- 13. Mark rejected flips (flip validation) ---
        let mark_rejected_flips_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mark_rejected_flips.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/mark_rejected_flips.wgsl").into()),
        });
        let mark_rejected_flips_params = GpuBuffers::create_params_buffer(device, [0, 0, 0, 0]);
        let mark_rejected_flips_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mark_rejected_flips_bgl"),
            entries: &[
                storage_rw_entry(0), storage_ro_entry(1), storage_ro_entry(2), storage_rw_entry(3),
                storage_rw_entry(4), storage_rw_entry(5), storage_rw_entry(6), uniform_entry(7),
            ],
        });
        let mark_rejected_flips_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mark_rejected_flips_bg"),
            layout: &mark_rejected_flips_bgl,
            entries: &[
                buf_entry(0, &bufs.act_tet_vec),
                buf_entry(1, &bufs.tet_opp),
                buf_entry(2, &bufs.tet_vote),
                buf_entry(3, &bufs.tet_info),
                buf_entry(4, &bufs.vote_arr),
                buf_entry(5, &bufs.flip_to_tet),
                buf_entry(6, &bufs.counters),
                buf_entry(7, &mark_rejected_flips_params),
            ],
        });
        let mark_rejected_flips_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mark_rejected_flips_pl"),
            bind_group_layouts: &[&mark_rejected_flips_bgl],
            push_constant_ranges: &[],
        });
        let mark_rejected_flips_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("mark_rejected_flips"),
            layout: Some(&mark_rejected_flips_pl),
            module: &mark_rejected_flips_shader,
            entry_point: Some("mark_rejected_flips"),
            compilation_options: Default::default(),
            cache: None,
        });

        // --- check_delaunay_fast (flip voting) ---
        let check_delaunay_fast_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("check_delaunay_fast.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/check_delaunay_fast.wgsl").into()),
        });
        let check_delaunay_fast_params = GpuBuffers::create_params_buffer(device, [0, 0, 0, 0]);
        let check_delaunay_fast_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("check_delaunay_fast_bgl"),
            entries: &[
                storage_rw_entry(0),  // act_tet_vec (read_write for queue management)
                storage_ro_entry(1),  // tets
                storage_rw_entry(2),  // tet_opp
                storage_ro_entry(3),  // tet_info
                storage_rw_entry(4),  // tet_vote_arr
                storage_rw_entry(5),  // vote_arr
                storage_rw_entry(6),  // counters
                storage_ro_entry(7),  // points
                uniform_entry(8),     // params
            ],
        });
        let check_delaunay_fast_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("check_delaunay_fast_bg"),
            layout: &check_delaunay_fast_bgl,
            entries: &[
                buf_entry(0, &bufs.act_tet_vec),
                buf_entry(1, &bufs.tets),
                buf_entry(2, &bufs.tet_opp),
                buf_entry(3, &bufs.tet_info),
                buf_entry(4, &bufs.tet_vote),  // tet_vote_arr
                buf_entry(5, &bufs.vote_arr),
                buf_entry(6, &bufs.counters),
                buf_entry(7, &bufs.points),
                buf_entry(8, &check_delaunay_fast_params),
            ],
        });
        let check_delaunay_fast_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("check_delaunay_fast_pl"),
            bind_group_layouts: &[&check_delaunay_fast_bgl],
            push_constant_ranges: &[],
        });
        let check_delaunay_fast_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("check_delaunay_fast"),
            layout: Some(&check_delaunay_fast_pl),
            module: &check_delaunay_fast_shader,
            entry_point: Some("check_delaunay_fast"),
            compilation_options: Default::default(),
            cache: None,
        });

        // --- check_delaunay_exact (flip voting with DD + SoS) ---
        let check_delaunay_exact_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("check_delaunay_exact.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/check_delaunay_exact.wgsl").into()),
        });
        let check_delaunay_exact_params = GpuBuffers::create_params_buffer(device, [0, 0, 0, 0]);
        let check_delaunay_exact_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("check_delaunay_exact_bgl"),
            entries: &[
                storage_rw_entry(0),  // act_tet_vec (read_write for queue management)
                storage_ro_entry(1),  // tets
                storage_rw_entry(2),  // tet_opp
                storage_ro_entry(3),  // tet_info
                storage_rw_entry(4),  // tet_vote_arr
                storage_rw_entry(5),  // vote_arr
                storage_rw_entry(6),  // counters
                storage_ro_entry(7),  // points
                uniform_entry(8),     // params
            ],
        });
        let check_delaunay_exact_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("check_delaunay_exact_bg"),
            layout: &check_delaunay_exact_bgl,
            entries: &[
                buf_entry(0, &bufs.act_tet_vec),
                buf_entry(1, &bufs.tets),
                buf_entry(2, &bufs.tet_opp),
                buf_entry(3, &bufs.tet_info),
                buf_entry(4, &bufs.tet_vote),  // tet_vote_arr
                buf_entry(5, &bufs.vote_arr),
                buf_entry(6, &bufs.counters),
                buf_entry(7, &bufs.points),
                buf_entry(8, &check_delaunay_exact_params),
            ],
        });
        let check_delaunay_exact_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("check_delaunay_exact_pl"),
            bind_group_layouts: &[&check_delaunay_exact_bgl],
            push_constant_ranges: &[],
        });
        let check_delaunay_exact_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("check_delaunay_exact"),
            layout: Some(&check_delaunay_exact_pl),
            module: &check_delaunay_exact_shader,
            entry_point: Some("check_delaunay_exact"),
            compilation_options: Default::default(),
            cache: None,
        });

        // --- allocate_flip23_slot ---
        let allocate_flip23_slot_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("allocate_flip23_slot.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/allocate_flip23_slot.wgsl").into()),
        });
        let allocate_flip23_slot_params = GpuBuffers::create_params_buffer(device, [0, 0, 0, 0]);
        let allocate_flip23_slot_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("allocate_flip23_slot_bgl"),
            entries: &[
                storage_ro_entry(0),  // flip_to_tet
                storage_ro_entry(1),  // tets
                storage_rw_entry(2),  // vert_free_arr
                storage_ro_entry(3),  // free_arr
                storage_rw_entry(4),  // flip23_new_slot
                uniform_entry(5),     // params
            ],
        });
        let allocate_flip23_slot_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("allocate_flip23_slot_bg"),
            layout: &allocate_flip23_slot_bgl,
            entries: &[
                buf_entry(0, &bufs.flip_to_tet),
                buf_entry(1, &bufs.tets),
                buf_entry(2, &bufs.vert_free_arr),
                buf_entry(3, &bufs.free_arr),
                buf_entry(4, &bufs.flip23_new_slot),
                buf_entry(5, &allocate_flip23_slot_params),
            ],
        });
        let allocate_flip23_slot_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("allocate_flip23_slot_pl"),
            bind_group_layouts: &[&allocate_flip23_slot_bgl],
            push_constant_ranges: &[],
        });
        let allocate_flip23_slot_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("allocate_flip23_slot"),
            layout: Some(&allocate_flip23_slot_pl),
            module: &allocate_flip23_slot_shader,
            entry_point: Some("allocate_flip23_slot"),
            compilation_options: Default::default(),
            cache: None,
        });

        // --- compact_if_negative ---
        let compact_if_negative_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("compact_if_negative.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/compact_if_negative.wgsl").into()),
        });
        let compact_if_negative_params = GpuBuffers::create_params_buffer(device, [0, 0, 0, 0]);
        let compact_if_negative_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("compact_if_negative_bgl"),
            entries: &[
                storage_ro_entry(0),  // input
                storage_rw_entry(1),  // output
                storage_rw_entry(2),  // counter
                uniform_entry(3),     // params
            ],
        });
        let compact_if_negative_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("compact_if_negative_bg"),
            layout: &compact_if_negative_bgl,
            entries: &[
                buf_entry(0, &bufs.vote_arr),      // input
                buf_entry(1, &bufs.flip_to_tet),   // output
                buf_entry(2, &bufs.counters),      // counter
                buf_entry(3, &compact_if_negative_params),
            ],
        });
        let compact_if_negative_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("compact_if_negative_pl"),
            bind_group_layouts: &[&compact_if_negative_bgl],
            push_constant_ranges: &[],
        });
        let compact_if_negative_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("compact_if_negative"),
            layout: Some(&compact_if_negative_pl),
            module: &compact_if_negative_shader,
            entry_point: Some("compact_if_negative"),
            compilation_options: Default::default(),
            cache: None,
        });

        // --- relocate_points_fast ---
        let relocate_points_fast_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("relocate_points_fast.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/relocate_points_fast.wgsl").into()),
        });
        let relocate_points_fast_params = GpuBuffers::create_params_buffer(device, [0, 0, 0, 0]);
        let relocate_points_fast_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("relocate_points_fast_bgl"),
            entries: &[
                storage_ro_entry(0),  // uninserted
                storage_rw_entry(1),  // vert_tet
                storage_ro_entry(2),  // tet_to_flip
                storage_ro_entry(3),  // flip_arr
                storage_ro_entry(4),  // points
                uniform_entry(5),     // params
            ],
        });
        let relocate_points_fast_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("relocate_points_fast_bg"),
            layout: &relocate_points_fast_bgl,
            entries: &[
                buf_entry(0, &bufs.uninserted),
                buf_entry(1, &bufs.vert_tet),
                buf_entry(2, &bufs.tet_to_flip),
                buf_entry(3, &bufs.flip_arr),
                buf_entry(4, &bufs.points),
                buf_entry(5, &relocate_points_fast_params),
            ],
        });
        let relocate_points_fast_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("relocate_points_fast_pl"),
            bind_group_layouts: &[&relocate_points_fast_bgl],
            push_constant_ranges: &[],
        });
        let relocate_points_fast_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("relocate_points_fast"),
            layout: Some(&relocate_points_fast_pl),
            module: &relocate_points_fast_shader,
            entry_point: Some("relocate_points_fast"),
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
            update_vert_free_pipeline,
            update_vert_free_bind_group,
            update_vert_free_params,
            mark_split_pipeline,
            mark_split_bind_group,
            mark_split_params,
            split_points_pipeline,
            split_points_bind_group,
            split_pipeline,
            split_bind_group,
            split_params,
            update_uninserted_vert_tet_pipeline,
            update_uninserted_vert_tet_bind_group,
            update_uninserted_vert_tet_params,
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
            check_delaunay_fast_pipeline,
            check_delaunay_fast_bind_group,
            check_delaunay_fast_params,
            check_delaunay_exact_pipeline,
            check_delaunay_exact_bind_group,
            check_delaunay_exact_params,
            allocate_flip23_slot_pipeline,
            allocate_flip23_slot_bind_group,
            allocate_flip23_slot_params,
            compact_if_negative_pipeline,
            compact_if_negative_bind_group,
            compact_if_negative_params,
            relocate_points_fast_pipeline,
            relocate_points_fast_bind_group,
            relocate_points_fast_params,
            update_opp_pipeline,
            update_opp_bind_group,
            update_opp_params,
            update_flip_trace_pipeline,
            update_flip_trace_bind_group,
            update_flip_trace_params,
            collect_free_slots_pipeline,
            collect_free_slots_bind_group,
            collect_free_slots_params,
            make_compact_map_pipeline,
            make_compact_map_bind_group,
            make_compact_map_params,
            compact_tets_pipeline,
            compact_tets_bind_group,
            compact_tets_params,
            mark_special_tets_pipeline,
            mark_special_tets_bind_group,
            mark_special_tets_params,
            mark_rejected_flips_pipeline,
            mark_rejected_flips_bind_group,
            mark_rejected_flips_params,
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
