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

    // Pipeline: negate inserted verts (mark successful insertions)
    pub negate_inserted_verts_pipeline: wgpu::ComputePipeline,
    pub negate_inserted_verts_bind_group: wgpu::BindGroup,
    pub negate_inserted_verts_params: wgpu::Buffer,

    // Pipeline: build insert list from tet_vert winners
    pub build_insert_list_pipeline: wgpu::ComputePipeline,
    pub build_insert_list_bind_group: wgpu::BindGroup,
    pub build_insert_list_params: wgpu::Buffer,

    // Pipeline: update vert free list (allocates tet blocks for inserting vertices)
    pub update_vert_free_pipeline: wgpu::ComputePipeline,
    pub update_vert_free_bind_group: wgpu::BindGroup,
    pub update_vert_free_params: wgpu::Buffer,

    // Pipeline: mark split (marks tets being split for concurrent detection)
    pub mark_split_pipeline: wgpu::ComputePipeline,
    pub mark_split_bind_group: wgpu::BindGroup,
    pub mark_split_params: wgpu::Buffer,

    // Pipeline: mark tet empty (sets empty flag on all tets before split)
    pub mark_tet_empty_pipeline: wgpu::ComputePipeline,
    pub mark_tet_empty_bind_group: wgpu::BindGroup,
    pub mark_tet_empty_params: wgpu::Buffer,

    // Pipeline: split points (updates vert_tet for vertices whose tets are splitting)
    pub split_points_pipeline: wgpu::ComputePipeline,
    pub split_points_bind_group: wgpu::BindGroup,
    pub split_points_params: wgpu::Buffer,

    // Pipeline: split tetra
    pub split_pipeline: wgpu::ComputePipeline,
    pub split_bind_group: wgpu::BindGroup,
    pub split_params: wgpu::Buffer,

    // Pipeline: split fixup (fixes adjacency for concurrent splits)
    pub split_fixup_pipeline: wgpu::ComputePipeline,
    pub split_fixup_bind_group: wgpu::BindGroup,
    pub split_fixup_params: wgpu::Buffer,

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

    // Pipeline: collect free slots (compaction)
    pub collect_free_slots_pipeline: wgpu::ComputePipeline,
    pub collect_free_slots_bind_group: wgpu::BindGroup,
    pub collect_free_slots_params: wgpu::Buffer,

    // Pipeline: make compact map (compaction)
    pub make_compact_map_pipeline: wgpu::ComputePipeline,
    pub make_compact_map_bind_group: wgpu::BindGroup,
    pub make_compact_map_params: wgpu::Buffer,

    // Pipeline: compact tets (compaction)
    pub compact_tets_pipeline: wgpu::ComputePipeline,
    pub compact_tets_bind_group: wgpu::BindGroup,
    pub compact_tets_params: wgpu::Buffer,

    // Pipeline: mark special tets (flip management)
    pub mark_special_tets_pipeline: wgpu::ComputePipeline,
    pub mark_special_tets_bind_group: wgpu::BindGroup,
    pub mark_special_tets_params: wgpu::Buffer,

    // Pipeline: update flip trace (flip management)
    pub update_flip_trace_pipeline: wgpu::ComputePipeline,
    pub update_flip_trace_bind_group: wgpu::BindGroup,
    pub update_flip_trace_params: wgpu::Buffer,

    // Pipeline: update block vert free list (block allocation)
    pub update_block_vert_free_list_pipeline: wgpu::ComputePipeline,
    pub update_block_vert_free_list_bind_group: wgpu::BindGroup,
    pub update_block_vert_free_list_params: wgpu::Buffer,

    // Pipeline: update block opp tet idx (block allocation)
    pub update_block_opp_tet_idx_pipeline: wgpu::ComputePipeline,
    pub update_block_opp_tet_idx_bind_group: wgpu::BindGroup,
    pub update_block_opp_tet_idx_params: wgpu::Buffer,

    // Pipeline: shift inf free idx (index shifting)
    pub shift_inf_free_idx_pipeline: wgpu::ComputePipeline,
    pub shift_inf_free_idx_bind_group: wgpu::BindGroup,
    pub shift_inf_free_idx_params: wgpu::Buffer,

    // Pipeline: update tet idx (index shifting)
    pub update_tet_idx_pipeline: wgpu::ComputePipeline,
    pub update_tet_idx_bind_group: wgpu::BindGroup,
    pub update_tet_idx_params: wgpu::Buffer,

    // Pipeline: shift opp tet idx (index shifting)
    pub shift_opp_tet_idx_pipeline: wgpu::ComputePipeline,
    pub shift_opp_tet_idx_bind_group: wgpu::BindGroup,
    pub shift_opp_tet_idx_params: wgpu::Buffer,

    // Pipeline: shift tet idx (index shifting)
    pub shift_tet_idx_pipeline: wgpu::ComputePipeline,
    pub shift_tet_idx_bind_group: wgpu::BindGroup,
    pub shift_tet_idx_params: wgpu::Buffer,

    // Pipeline: make reverse map (utility)
    pub make_reverse_map_pipeline: wgpu::ComputePipeline,
    pub make_reverse_map_bind_group: wgpu::BindGroup,
    pub make_reverse_map_params: wgpu::Buffer,

    // Pipeline: update opp (CRITICAL - flip adjacency updates)
    pub update_opp_pipeline: wgpu::ComputePipeline,
    pub update_opp_bind_group: wgpu::BindGroup,
    pub update_opp_params: wgpu::Buffer,

    // Pipeline: mark rejected flips (flip validation)
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

        // params: x = num_uninserted, y = insertion_rule (0=circumcenter, 1=centroid, 2=random)
        let vote_params = GpuBuffers::create_params_buffer(device, [num_points, 0, 0, 0]);

        let vote_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vote_bgl"),
            entries: &[
                storage_ro_entry(0),  // points
                storage_ro_entry(1),  // uninserted (vertexArr in original)
                storage_ro_entry(2),  // vert_tet (vertexTetArr in original)
                storage_ro_entry(3),  // tets
                storage_rw_entry(4),  // vert_sphere (vertSphereArr - atomic i32)
                storage_rw_entry(5),  // tet_sphere (tetSphereArr - atomic i32)
                uniform_entry(6),     // params
            ],
        });

        let vote_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vote_bg"),
            layout: &vote_bgl,
            entries: &[
                buf_entry(0, &bufs.points),
                buf_entry(1, &bufs.uninserted),
                buf_entry(2, &bufs.vert_tet),
                buf_entry(3, &bufs.tets),
                buf_entry(4, &bufs.vert_sphere),
                buf_entry(5, &bufs.tet_sphere),
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

        // params: x = num_uninserted
        let pick_params = GpuBuffers::create_params_buffer(device, [num_points, 0, 0, 0]);

        let pick_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pick_bgl"),
            entries: &[
                storage_ro_entry(0),  // uninserted (vertexArr in original)
                storage_ro_entry(1),  // vert_tet (vertexTetArr in original)
                storage_rw_entry(2),  // vert_sphere (vertSphereArr - atomic i32, need RW for atomicLoad)
                storage_rw_entry(3),  // tet_sphere (tetSphereArr - atomic i32, need RW for atomicLoad)
                storage_rw_entry(4),  // tet_vert (tetVertArr - atomic i32)
                uniform_entry(5),     // params
            ],
        });

        let pick_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pick_bg"),
            layout: &pick_bgl,
            entries: &[
                buf_entry(0, &bufs.uninserted),
                buf_entry(1, &bufs.vert_tet),
                buf_entry(2, &bufs.vert_sphere),
                buf_entry(3, &bufs.tet_sphere),
                buf_entry(4, &bufs.tet_vert),
                buf_entry(5, &pick_params),
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

        // --- Negate inserted verts pipeline ---
        let negate_inserted_verts_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("negate_inserted_verts.wgsl"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/negate_inserted_verts.wgsl").into(),
            ),
        });

        let negate_inserted_verts_params = GpuBuffers::create_params_buffer(device, [0, 0, 0, 0]);

        let negate_inserted_verts_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("negate_inserted_verts_bgl"),
            entries: &[
                storage_rw_entry(0),  // vert_tet
                storage_ro_entry(1),  // tet_vert
                uniform_entry(2),     // params
            ],
        });

        let negate_inserted_verts_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("negate_inserted_verts_bg"),
            layout: &negate_inserted_verts_bgl,
            entries: &[
                buf_entry(0, &bufs.vert_tet),
                buf_entry(1, &bufs.tet_vert),
                buf_entry(2, &negate_inserted_verts_params),
            ],
        });

        let negate_inserted_verts_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("negate_inserted_verts_pl"),
            bind_group_layouts: &[&negate_inserted_verts_bgl],
            push_constant_ranges: &[],
        });

        let negate_inserted_verts_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("negate_inserted_verts"),
            layout: Some(&negate_inserted_verts_pl),
            module: &negate_inserted_verts_shader,
            entry_point: Some("negate_inserted_verts"),
            compilation_options: Default::default(),
            cache: None,
        });

        // --- Build insert list pipeline ---
        let build_insert_list_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("build_insert_list.wgsl"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/build_insert_list.wgsl").into(),
            ),
        });

        let build_insert_list_params = GpuBuffers::create_params_buffer(device, [max_tets, 0, 0, 0]);

        let build_insert_list_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("build_insert_list_bgl"),
            entries: &[
                storage_ro_entry(0),  // tet_info
                storage_rw_entry(1),  // tet_vert (atomic i32, needs RW)
                storage_ro_entry(2),  // uninserted
                storage_rw_entry(3),  // insert_list
                storage_rw_entry(4),  // counters
                uniform_entry(5),     // params
            ],
        });

        let build_insert_list_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("build_insert_list_bg"),
            layout: &build_insert_list_bgl,
            entries: &[
                buf_entry(0, &bufs.tet_info),
                buf_entry(1, &bufs.tet_vert),
                buf_entry(2, &bufs.uninserted),
                buf_entry(3, &bufs.insert_list),
                buf_entry(4, &bufs.counters),
                buf_entry(5, &build_insert_list_params),
            ],
        });

        let build_insert_list_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("build_insert_list_pl"),
            bind_group_layouts: &[&build_insert_list_bgl],
            push_constant_ranges: &[],
        });

        let build_insert_list_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("build_insert_list"),
            layout: Some(&build_insert_list_pl),
            module: &build_insert_list_shader,
            entry_point: Some("build_insert_list"),
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
                storage_ro_entry(4),  // uninserted (to convert position→vertex)
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
                buf_entry(4, &bufs.uninserted),
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

        // --- Mark tet empty pipeline ---
        let mark_tet_empty_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mark_tet_empty.wgsl"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/mark_tet_empty.wgsl").into(),
            ),
        });

        let mark_tet_empty_params = GpuBuffers::create_params_buffer(device, [0, 0, 0, 0]);

        let mark_tet_empty_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mark_tet_empty_bgl"),
            entries: &[
                storage_rw_entry(0),  // tet_info
                uniform_entry(1),     // params
            ],
        });

        let mark_tet_empty_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mark_tet_empty_bg"),
            layout: &mark_tet_empty_bgl,
            entries: &[
                buf_entry(0, &bufs.tet_info),
                buf_entry(1, &mark_tet_empty_params),
            ],
        });

        let mark_tet_empty_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mark_tet_empty_pl"),
            bind_group_layouts: &[&mark_tet_empty_bgl],
            push_constant_ranges: &[],
        });

        let mark_tet_empty_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("mark_tet_empty"),
            layout: Some(&mark_tet_empty_pl),
            module: &mark_tet_empty_shader,
            entry_point: Some("mark_tet_empty"),
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

        let split_points_params = GpuBuffers::create_params_buffer(device, [0, 0, 0, 0]);

        let split_points_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("split_points_bgl"),
            entries: &[
                storage_ro_entry(0),  // points
                storage_ro_entry(1),  // uninserted
                storage_rw_entry(2),  // vert_tet
                storage_ro_entry(3),  // tet_to_vert
                storage_ro_entry(4),  // tets
                storage_ro_entry(5),  // free_arr
                uniform_entry(6),     // params
            ],
        });

        let split_points_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("split_points_bg"),
            layout: &split_points_bgl,
            entries: &[
                buf_entry(0, &bufs.points),
                buf_entry(1, &bufs.uninserted),
                buf_entry(2, &bufs.vert_tet),
                buf_entry(3, &bufs.tet_to_vert),
                buf_entry(4, &bufs.tets),
                buf_entry(5, &bufs.free_arr),
                buf_entry(6, &split_points_params),
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
            entry_point: Some("split_points"),
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
                storage_rw_entry(11), // breadcrumbs (debug)
                storage_rw_entry(12), // thread_debug (debug)
                storage_ro_entry(13), // uninserted (to get vertex from position)
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
                buf_entry(11, &bufs.breadcrumbs),
                buf_entry(12, &bufs.thread_debug),
                buf_entry(13, &bufs.uninserted),
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

        // --- Split fixup pipeline ---
        let split_fixup_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("split_fixup.wgsl"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/split_fixup.wgsl").into(),
            ),
        });

        let split_fixup_params = GpuBuffers::create_params_buffer(device, [0, 0, 0, 0]);

        let split_fixup_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("split_fixup_bgl"),
            entries: &[
                storage_ro_entry(0),  // insert_list
                storage_ro_entry(1),  // tet_to_vert
                storage_ro_entry(2),  // tet_split_map
                storage_rw_entry(3),  // tet_opp
                uniform_entry(4),     // params
            ],
        });

        let split_fixup_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("split_fixup_bg"),
            layout: &split_fixup_bgl,
            entries: &[
                buf_entry(0, &bufs.insert_list),
                buf_entry(1, &bufs.tet_to_vert),
                buf_entry(2, &bufs.tet_split_map),
                buf_entry(3, &bufs.tet_opp),
                buf_entry(4, &split_fixup_params),
            ],
        });

        let split_fixup_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("split_fixup_pl"),
            bind_group_layouts: &[&split_fixup_bgl],
            push_constant_ranges: &[],
        });

        let split_fixup_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("split_fixup"),
            layout: Some(&split_fixup_pl),
            module: &split_fixup_shader,
            entry_point: Some("fixup_split_adjacency"),
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
                storage_rw_entry(4),  // update_debug
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
                buf_entry(4, &bufs.update_debug),
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

        // params: x = num_uninserted, y = max_tets
        let reset_votes_params = GpuBuffers::create_params_buffer(device, [num_points, max_tets, 0, 0]);

        let reset_votes_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("reset_votes_bgl"),
            entries: &[
                storage_rw_entry(0),  // vert_sphere
                storage_rw_entry(1),  // tet_sphere
                storage_rw_entry(2),  // tet_vert
                uniform_entry(3),     // params
            ],
        });

        let reset_votes_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("reset_votes_bg"),
            layout: &reset_votes_bgl,
            entries: &[
                buf_entry(0, &bufs.vert_sphere),
                buf_entry(1, &bufs.tet_sphere),
                buf_entry(2, &bufs.tet_vert),
                buf_entry(3, &reset_votes_params),
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
            negate_inserted_verts_pipeline,
            negate_inserted_verts_bind_group,
            negate_inserted_verts_params,
            build_insert_list_pipeline,
            build_insert_list_bind_group,
            build_insert_list_params,
            update_vert_free_pipeline,
            update_vert_free_bind_group,
            update_vert_free_params,
            mark_split_pipeline,
            mark_split_bind_group,
            mark_split_params,
            mark_tet_empty_pipeline,
            mark_tet_empty_bind_group,
            mark_tet_empty_params,
            split_points_pipeline,
            split_points_bind_group,
            split_points_params,
            split_pipeline,
            split_bind_group,
            split_params,
            split_fixup_pipeline,
            split_fixup_bind_group,
            split_fixup_params,
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
