use super::buffers::GpuBuffers;

/// All compute pipelines and their bind groups.
pub struct Pipelines {
    // Pipeline: init (make_first_tetra)
    pub init_pipeline: wgpu::ComputePipeline,
    // init_bind_group removed - now created dynamically at dispatch time to avoid 128 MB limit
    pub init_params: wgpu::Buffer,

    // Pipeline: init_vert_tet (OBSOLETE - vert_tet now initialized on CPU)
    // Keeping these fields to avoid breaking existing code, but shader is no longer dispatched
    pub init_vert_tet_pipeline: wgpu::ComputePipeline,
    pub init_vert_tet_bind_group: wgpu::BindGroup,

    // Pipeline: vote for point
    pub vote_pipeline: wgpu::ComputePipeline,
    // vote_bind_group removed - now created dynamically at dispatch time
    pub vote_params: wgpu::Buffer,

    // Pipeline: pick winner
    pub pick_pipeline: wgpu::ComputePipeline,
    pub pick_bind_group: wgpu::BindGroup,
    pub pick_params: wgpu::Buffer,

    // Pipeline: build insert list (filters exact winners after pick_winner)
    pub build_insert_list_pipeline: wgpu::ComputePipeline,
    pub build_insert_list_bind_group: wgpu::BindGroup,
    pub build_insert_list_params: wgpu::Buffer,

    // Pipeline: update vert free list (allocates tet blocks for inserting vertices)
    pub update_vert_free_pipeline: wgpu::ComputePipeline,
    pub update_vert_free_bind_group: wgpu::BindGroup,
    pub update_vert_free_params: wgpu::Buffer,

    // Pipeline: mark split - REMOVED (obsolete, pick_winner_point populates tet_to_vert)

    // Pipeline: split points (updates vert_tet for uninserted vertices whose tets are splitting)
    pub split_points_pipeline: wgpu::ComputePipeline,
    // split_points_bind_group removed - now created dynamically at dispatch time
    pub split_points_params: wgpu::Buffer,

    // Pipeline: split tetra
    pub split_pipeline: wgpu::ComputePipeline,
    // split_bind_group removed - now created dynamically at dispatch time
    pub split_params: wgpu::Buffer,

    // Pipeline: update uninserted vert_tet (fixes dead tet references)
    pub update_uninserted_vert_tet_pipeline: wgpu::ComputePipeline,
    pub update_uninserted_vert_tet_bind_group: wgpu::BindGroup,
    pub update_uninserted_vert_tet_params: wgpu::Buffer,

    // Pipeline: flip vote (phase 1 - marks violations via atomicMin)
    pub flip_vote_pipeline: wgpu::ComputePipeline,
    // Pipeline: flip check (phase 2 - executes flips if vote won)
    pub flip_pipeline: wgpu::ComputePipeline,
    // flip_bind_group and flip_bind_group_b removed - now created dynamically at dispatch time
    pub flip_params: wgpu::Buffer,

    // Pipeline: reset votes
    pub reset_votes_pipeline: wgpu::ComputePipeline,
    pub reset_votes_bind_group: wgpu::BindGroup,
    pub reset_votes_params: wgpu::Buffer,

    // Pipeline: reset flip votes (atomicMin, INT_MAX sentinel)
    pub reset_flip_votes_pipeline: wgpu::ComputePipeline,

    // Pipeline: gather failed
    pub gather_pipeline: wgpu::ComputePipeline,
    // gather_bind_group removed - now created dynamically at dispatch time
    pub gather_params: wgpu::Buffer,

    // Pipeline: check delaunay fast (flip voting)
    pub check_delaunay_fast_pipeline: wgpu::ComputePipeline,
    // check_delaunay_fast_bind_group removed - now created dynamically at dispatch time
    pub check_delaunay_fast_params: wgpu::Buffer,

    // Pipeline: check delaunay exact (flip voting with DD + SoS)
    pub check_delaunay_exact_pipeline: wgpu::ComputePipeline,
    // check_delaunay_exact_bind_group removed - now created dynamically at dispatch time
    pub check_delaunay_exact_params: wgpu::Buffer,

    // Pipeline: allocate flip23 slot
    pub allocate_flip23_slot_pipeline: wgpu::ComputePipeline,
    pub allocate_flip23_slot_params: wgpu::Buffer,

    // Pipeline: compact if negative
    pub compact_if_negative_pipeline: wgpu::ComputePipeline,
    pub compact_if_negative_bind_group: wgpu::BindGroup,
    pub compact_if_negative_params: wgpu::Buffer,

    // Pipeline: compact vertex arrays (atomic-based, 2 passes - fast path)
    pub compact_atomic_pipeline: wgpu::ComputePipeline,
    pub compact_atomic_bind_group: wgpu::BindGroup,
    pub compact_atomic_params: wgpu::Buffer,

    // Pipeline: zero compaction flags (GPU zeroing instead of CPU→GPU transfer)
    pub zero_compaction_flags_pipeline: wgpu::ComputePipeline,
    pub zero_compaction_flags_bind_group: wgpu::BindGroup,

    // Pipeline: compact vertex arrays (prefix-sum-based, 3 passes)
    pub compact_mark_inserted_pipeline: wgpu::ComputePipeline,
    pub compact_mark_inserted_bind_group: wgpu::BindGroup,
    pub compact_invert_flags_pipeline: wgpu::ComputePipeline,
    pub compact_invert_flags_bind_group: wgpu::BindGroup,
    pub compact_scatter_pipeline: wgpu::ComputePipeline,
    pub compact_scatter_bind_group: wgpu::BindGroup,
    pub compact_vertex_arrays_params: wgpu::Buffer,

    // Pipelines: pack/unpack for compaction prefix sum
    pub pack_flags_pipeline: wgpu::ComputePipeline,
    pub pack_flags_bind_group: wgpu::BindGroup,
    pub pack_flags_params: wgpu::Buffer,
    pub unpack_compact_pipeline: wgpu::ComputePipeline,
    pub unpack_compact_bind_group: wgpu::BindGroup,
    pub unpack_compact_params: wgpu::Buffer,
    pub inclusive_to_exclusive_pipeline: wgpu::ComputePipeline,
    pub inclusive_to_exclusive_bind_group: wgpu::BindGroup,
    pub inclusive_to_exclusive_params: wgpu::Buffer,

    // Pipeline: init tet_to_flip buffer (GPU initialization instead of CPU→GPU transfer)
    pub init_tet_to_flip_pipeline: wgpu::ComputePipeline,
    pub init_tet_to_flip_bind_group: wgpu::BindGroup,
    pub init_tet_to_flip_params: wgpu::Buffer,

    // Pipeline: relocate points fast
    pub relocate_points_fast_pipeline: wgpu::ComputePipeline,
    pub relocate_points_fast_params: wgpu::Buffer,

    // Pipeline: update opp (fix adjacency after flips)
    pub update_opp_pipeline: wgpu::ComputePipeline,
    pub update_opp_params: wgpu::Buffer,

    // Pipeline: donate freed tets (post-flip: 3-2 freed tets → free list)
    pub donate_freed_tets_pipeline: wgpu::ComputePipeline,
    pub donate_freed_tets_bgl: wgpu::BindGroupLayout,
    pub donate_freed_tets_params: wgpu::Buffer,

    // Pipeline: update flip trace (build flip history chains for relocateAll)
    pub update_flip_trace_pipeline: wgpu::ComputePipeline,
    pub update_flip_trace_params: wgpu::Buffer,

    // Bind group layouts (exposed for dynamic partial binding)
    pub update_flip_trace_bgl: wgpu::BindGroupLayout,
    pub update_opp_bgl: wgpu::BindGroupLayout,
    pub relocate_points_fast_bgl: wgpu::BindGroupLayout,

    // Phase 2: Additional layouts for 11 pipelines that bind large buffers
    pub init_bgl: wgpu::BindGroupLayout,
    pub vote_bgl: wgpu::BindGroupLayout,
    pub split_points_bgl: wgpu::BindGroupLayout,
    pub split_bgl: wgpu::BindGroupLayout,
    pub flip_bgl: wgpu::BindGroupLayout,
    pub gather_bgl: wgpu::BindGroupLayout,
    pub compact_tets_bgl: wgpu::BindGroupLayout,
    pub mark_special_tets_bgl: wgpu::BindGroupLayout,
    pub check_delaunay_fast_bgl: wgpu::BindGroupLayout,
    pub check_delaunay_exact_bgl: wgpu::BindGroupLayout,
    pub mark_rejected_flips_bgl: wgpu::BindGroupLayout,
    pub allocate_flip23_slot_bgl: wgpu::BindGroupLayout,

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
    // compact_tets_bind_group removed - now created dynamically at dispatch time
    pub compact_tets_params: wgpu::Buffer,

    // Pipeline: mark special tets (between fast/exact flipping)
    pub mark_special_tets_pipeline: wgpu::ComputePipeline,
    // mark_special_tets_bind_group removed - now created dynamically at dispatch time
    pub mark_special_tets_params: wgpu::Buffer,

    // Pipeline: repair adjacency (post-flip back-pointer fix)
    pub repair_adjacency_pipeline: wgpu::ComputePipeline,
    pub repair_adjacency_bgl: wgpu::BindGroupLayout,
    pub repair_adjacency_params: wgpu::Buffer,

    // Pipeline: collect active tets (populate act_tet_vec before flipping)
    pub collect_active_tets_pipeline: wgpu::ComputePipeline,
    pub collect_active_tets_bgl: wgpu::BindGroupLayout,
    pub collect_active_tets_params: wgpu::Buffer,

    // Pipeline: mark rejected flips
    pub mark_rejected_flips_pipeline: wgpu::ComputePipeline,
    pub mark_rejected_flips_params: wgpu::Buffer,

    // --- GPU Prefix Sum Pipelines ---
    pub transform_pipeline: wgpu::ComputePipeline,
    pub transform_bind_group: wgpu::BindGroup,
    pub transform_params: wgpu::Buffer,

    pub reduce_pipeline: wgpu::ComputePipeline,
    pub reduce_bind_group: wgpu::BindGroup,

    pub spine_scan_pipeline: wgpu::ComputePipeline,
    pub spine_scan_bind_group: wgpu::BindGroup,

    pub downsweep_pipeline: wgpu::ComputePipeline,
    pub downsweep_bind_group: wgpu::BindGroup,

    pub unpack_pipeline: wgpu::ComputePipeline,
    pub unpack_bind_group: wgpu::BindGroup,
    pub unpack_params: wgpu::Buffer,
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

        // init_bind_group removed - created dynamically at dispatch time to avoid 128 MB buffer binding limit

        let init_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("init_pl"),
            bind_group_layouts: &[&init_bgl],
            immediate_size: 0,
        });

        let init_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("init"),
            layout: Some(&init_pl),
            module: &init_shader,
            entry_point: Some("make_first_tetra"),
            compilation_options: Default::default(),
            cache: None,
        });

        // --- Init vert_tet pipeline (parallel initialization) ---
        // Reuse init_bgl since the shader module declares all bindings globally
        let init_vert_tet_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("init_vert_tet_bg"),
            layout: &init_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: bufs.points.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: bufs.tets.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bufs.tet_opp.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: bufs.tet_info.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: bufs.vert_tet.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: bufs.free_arr.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: bufs.vert_free_arr.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: bufs.counters.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: init_params.as_entire_binding(),
                },
            ],
        });

        let init_vert_tet_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("init_vert_tet_pl"),
            bind_group_layouts: &[&init_bgl],
            immediate_size: 0,
        });

        let init_vert_tet_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("init_vert_tet"),
            layout: Some(&init_vert_tet_pl),
            module: &init_shader,
            entry_point: Some("init_vert_tet"),
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
                storage_rw_entry(3),  // tet_vote (atomic i32)
                storage_rw_entry(4),  // vert_sphere (i32, stores sphere values)
                storage_ro_entry(5),  // vert_tet
                storage_ro_entry(6),  // uninserted
                uniform_entry(7),     // params
            ],
        });

        // vote_bind_group removed - created dynamically at dispatch time

        let vote_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("vote_pl"),
            bind_group_layouts: &[&vote_bgl],
            immediate_size: 0,
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
        // Uses pick_winner_point entry point from vote.wgsl
        let pick_params = GpuBuffers::create_params_buffer(device, [num_points, 0, 0, 0]);

        let pick_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pick_bgl"),
            entries: &[
                storage_ro_entry(0),  // vert_sphere
                storage_ro_entry(1),  // tet_vote (winning sphere values)
                storage_ro_entry(2),  // vert_tet
                storage_rw_entry(3),  // tet_to_vert (atomicMin winner selection)
                storage_ro_entry(4),  // uninserted (map position to vertex ID)
                storage_ro_entry(5),  // tet_info (alive state check)
                uniform_entry(6),     // params
            ],
        });

        let pick_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pick_bg"),
            layout: &pick_bgl,
            entries: &[
                buf_entry(0, &bufs.vert_sphere),
                buf_entry(1, &bufs.tet_vote),
                buf_entry(2, &bufs.vert_tet),
                buf_entry(3, &bufs.tet_to_vert),
                buf_entry(4, &bufs.uninserted),
                buf_entry(5, &bufs.tet_info),
                buf_entry(6, &pick_params),
            ],
        });

        let pick_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pick_pl"),
            bind_group_layouts: &[&pick_bgl],
            immediate_size: 0,
        });

        let pick_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pick_winner"),
            layout: Some(&pick_pl),
            module: &vote_shader,  // Use vote.wgsl which contains pick_winner_point
            entry_point: Some("pick_winner_point"),
            compilation_options: Default::default(),
            cache: None,
        });

        // --- Build insert list pipeline ---
        // Second pass: filters exact winners after pick_winner (prevents duplicates)
        let build_insert_list_params = GpuBuffers::create_params_buffer(device, [0, 0, 0, 0]);

        let build_insert_list_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("build_insert_list_bgl"),
            entries: &[
                storage_ro_entry(0),  // vert_tet
                storage_ro_entry(1),  // tet_to_vert (winner indices from pick_winner)
                storage_ro_entry(2),  // tet_info
                storage_rw_entry(3),  // insert_list
                storage_rw_entry(4),  // counters
                storage_ro_entry(5),  // uninserted (map position to vertex ID)
                uniform_entry(6),     // params
            ],
        });

        let build_insert_list_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("build_insert_list_bg"),
            layout: &build_insert_list_bgl,
            entries: &[
                buf_entry(0, &bufs.vert_tet),
                buf_entry(1, &bufs.tet_to_vert),
                buf_entry(2, &bufs.tet_info),
                buf_entry(3, &bufs.insert_list),
                buf_entry(4, &bufs.counters),
                buf_entry(5, &bufs.uninserted),
                buf_entry(6, &build_insert_list_params),
            ],
        });

        let build_insert_list_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("build_insert_list_pl"),
            bind_group_layouts: &[&build_insert_list_bgl],
            immediate_size: 0,
        });

        let build_insert_list_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("build_insert_list"),
            layout: Some(&build_insert_list_pl),
            module: &vote_shader,  // Use vote.wgsl which contains build_insert_list
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
                storage_ro_entry(3),  // uninserted
                uniform_entry(4),     // params (x = num_insertions, y = start_free_idx)
            ],
        });

        let update_vert_free_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("update_vert_free_bg"),
            layout: &update_vert_free_bgl,
            entries: &[
                buf_entry(0, &bufs.insert_list),
                buf_entry(1, &bufs.vert_free_arr),
                buf_entry(2, &bufs.free_arr),
                buf_entry(3, &bufs.uninserted),
                buf_entry(4, &update_vert_free_params),
            ],
        });

        let update_vert_free_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("update_vert_free_pl"),
            bind_group_layouts: &[&update_vert_free_bgl],
            immediate_size: 0,
        });

        let update_vert_free_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("update_vert_free_list"),
            layout: Some(&update_vert_free_pl),
            module: &update_vert_free_shader,
            entry_point: Some("update_vert_free_list"),
            compilation_options: Default::default(),
            cache: None,
        });

        // --- Mark split pipeline - REMOVED (obsolete) ---

        // --- Split points pipeline ---
        let split_points_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("split_points.wgsl"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/split_points.wgsl").into(),
            ),
        });

        let split_points_params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("split_points_params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
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
                uniform_entry(10),    // params
            ],
        });

        // split_points_bind_group removed - created dynamically at dispatch time

        let split_points_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("split_points_pl"),
            bind_group_layouts: &[&split_points_bgl],
            immediate_size: 0,
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
                storage_ro_entry(11), // block_owner (pre-computed block ownership)
                storage_ro_entry(12), // uninserted (position -> vertex ID mapping)
                // NOTE: Removed breadcrumbs and thread_debug to stay under 10 storage buffer limit
            ],
        });

        // split_bind_group removed - created dynamically at dispatch time

        let split_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("split_pl"),
            bind_group_layouts: &[&split_bgl],
            immediate_size: 0,
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
                storage_ro_entry(4),  // tet_opp (for adjacency walking)
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
                buf_entry(4, &bufs.tet_opp),
            ],
        });

        let update_uninserted_vert_tet_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("update_uninserted_vert_tet_pl"),
            bind_group_layouts: &[&update_uninserted_vert_tet_bgl],
            immediate_size: 0,
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
                storage_ro_entry(11), // block_owner
                storage_rw_entry(12), // tet_vote (for flip voting)
                storage_rw_entry(13), // tet_msg_arr
                storage_rw_entry(14), // flip_arr
                storage_rw_entry(15), // encoded_face_vi_arr
            ],
        });

        // flip_bind_group and flip_bind_group_b removed - created dynamically at dispatch time

        let flip_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("flip_pl"),
            bind_group_layouts: &[&flip_bgl],
            immediate_size: 0,
        });

        let flip_vote_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("flip_vote"),
            layout: Some(&flip_pl),
            module: &flip_shader,
            entry_point: Some("flip_vote"),
            compilation_options: Default::default(),
            cache: None,
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

        let reset_votes_params = GpuBuffers::create_params_buffer(device, [max_tets, num_points, 0, 0]);

        let reset_votes_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("reset_votes_bgl"),
            entries: &[
                storage_rw_entry(0),  // tet_vote
                storage_rw_entry(1),  // vert_sphere
                storage_rw_entry(2),  // tet_to_vert (NEW: reset winner array)
                uniform_entry(3),     // params (max_tets, num_uninserted)
            ],
        });

        let reset_votes_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("reset_votes_bg"),
            layout: &reset_votes_bgl,
            entries: &[
                buf_entry(0, &bufs.tet_vote),
                buf_entry(1, &bufs.vert_sphere),
                buf_entry(2, &bufs.tet_to_vert),
                buf_entry(3, &reset_votes_params),
            ],
        });

        let reset_votes_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("reset_votes_pl"),
            bind_group_layouts: &[&reset_votes_bgl],
            immediate_size: 0,
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

        let reset_flip_votes_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("reset_flip_votes"),
                layout: Some(&reset_votes_pl),
                module: &reset_votes_shader,
                entry_point: Some("reset_flip_votes"),
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

        // gather_bind_group removed - created dynamically at dispatch time

        let gather_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gather_pl"),
            bind_group_layouts: &[&gather_bgl],
            immediate_size: 0,
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
            immediate_size: 0,
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
            immediate_size: 0,
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
        // compact_tets_bind_group removed - created dynamically at dispatch time
        let compact_tets_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("compact_tets_pl"),
            bind_group_layouts: &[&compact_tets_bgl],
            immediate_size: 0,
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
        // mark_special_tets_bind_group removed - created dynamically at dispatch time
        let mark_special_tets_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mark_special_tets_pl"),
            bind_group_layouts: &[&mark_special_tets_bgl],
            immediate_size: 0,
        });
        let mark_special_tets_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("mark_special_tets"),
            layout: Some(&mark_special_tets_pl),
            module: &mark_special_tets_shader,
            entry_point: Some("mark_special_tets"),
            compilation_options: Default::default(),
            cache: None,
        });

        // --- Collect active tets (populate act_tet_vec before flipping) ---
        // --- Repair adjacency pipeline ---
        let repair_adjacency_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("repair_adjacency.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/repair_adjacency.wgsl").into()),
        });
        let repair_adjacency_params = GpuBuffers::create_params_buffer(device, [0, 0, 0, 0]);
        let repair_adjacency_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("repair_adjacency_bgl"),
            entries: &[
                storage_ro_entry(0),  // tets
                storage_rw_entry(1),  // tet_opp
                storage_ro_entry(2),  // tet_info
                uniform_entry(3),     // params
            ],
        });
        let repair_adjacency_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("repair_adjacency_pl"),
            bind_group_layouts: &[&repair_adjacency_bgl],
            immediate_size: 0,
        });
        let repair_adjacency_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("repair_adjacency"),
            layout: Some(&repair_adjacency_pl),
            module: &repair_adjacency_shader,
            entry_point: Some("repair_adjacency"),
            compilation_options: Default::default(),
            cache: None,
        });

        let collect_active_tets_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("collect_active_tets.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/collect_active_tets.wgsl").into()),
        });
        let collect_active_tets_params = GpuBuffers::create_params_buffer(device, [0, 0, 0, 0]);
        let collect_active_tets_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("collect_active_tets_bgl"),
            entries: &[
                storage_ro_entry(0),  // tet_info
                storage_rw_entry(1),  // act_tet_vec
                storage_rw_entry(2),  // counters
                uniform_entry(3),     // params
            ],
        });
        let collect_active_tets_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("collect_active_tets_pl"),
            bind_group_layouts: &[&collect_active_tets_bgl],
            immediate_size: 0,
        });
        let collect_active_tets_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("collect_active_tets"),
            layout: Some(&collect_active_tets_pl),
            module: &collect_active_tets_shader,
            entry_point: Some("collect_active_tets"),
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
        let update_flip_trace_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("update_flip_trace_pl"),
            bind_group_layouts: &[&update_flip_trace_bgl],
            immediate_size: 0,
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
        let update_opp_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("update_opp_pl"),
            bind_group_layouts: &[&update_opp_bgl],
            immediate_size: 0,
        });
        let update_opp_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("update_opp"),
            layout: Some(&update_opp_pl),
            module: &update_opp_shader,
            entry_point: Some("update_opp"),
            compilation_options: Default::default(),
            cache: None,
        });

        // --- 12b. Donate freed tets (post-flip: return 3-2 freed tets to free list) ---
        let donate_freed_tets_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("donate_freed_tets.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/donate_freed_tets.wgsl").into()),
        });
        let donate_freed_tets_params = GpuBuffers::create_params_buffer(device, [0, 0, 0, 0]);
        let donate_freed_tets_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("donate_freed_tets_bgl"),
            entries: &[
                storage_ro_entry(0),  // flip_arr (read-only)
                storage_rw_entry(1),  // free_arr (read-write)
                storage_rw_entry(2),  // vert_free_arr (atomic)
                storage_ro_entry(3),  // block_owner (read-only)
                uniform_entry(4),     // params
            ],
        });
        let donate_freed_tets_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("donate_freed_tets_pl"),
            bind_group_layouts: &[&donate_freed_tets_bgl],
            immediate_size: 0,
        });
        let donate_freed_tets_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("donate_freed_tets"),
            layout: Some(&donate_freed_tets_pl),
            module: &donate_freed_tets_shader,
            entry_point: Some("donate_freed_tets"),
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
        let mark_rejected_flips_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mark_rejected_flips_pl"),
            bind_group_layouts: &[&mark_rejected_flips_bgl],
            immediate_size: 0,
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
        // check_delaunay_fast_bind_group removed - created dynamically at dispatch time
        let check_delaunay_fast_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("check_delaunay_fast_pl"),
            bind_group_layouts: &[&check_delaunay_fast_bgl],
            immediate_size: 0,
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
        // check_delaunay_exact_bind_group removed - created dynamically at dispatch time
        let check_delaunay_exact_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("check_delaunay_exact_pl"),
            bind_group_layouts: &[&check_delaunay_exact_bgl],
            immediate_size: 0,
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
        let allocate_flip23_slot_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("allocate_flip23_slot_pl"),
            bind_group_layouts: &[&allocate_flip23_slot_bgl],
            immediate_size: 0,
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
            immediate_size: 0,
        });
        let compact_if_negative_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("compact_if_negative"),
            layout: Some(&compact_if_negative_pl),
            module: &compact_if_negative_shader,
            entry_point: Some("compact_if_negative"),
            compilation_options: Default::default(),
            cache: None,
        });

        // --- compact_vertex_arrays_atomic (fast path for small datasets, 2 passes) ---
        let compact_atomic_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("compact_vertex_arrays_atomic.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/compact_vertex_arrays_atomic.wgsl").into()),
        });
        let compact_atomic_params = GpuBuffers::create_params_buffer(device, [0, 0, 0, 0]);

        let compact_atomic_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("compact_atomic_bgl"),
            entries: &[
                storage_ro_entry(0),  // insert_list
                storage_ro_entry(1),  // uninserted_in
                storage_ro_entry(2),  // vert_tet_in
                storage_rw_entry(3),  // uninserted_out
                storage_rw_entry(4),  // vert_tet_out
                storage_rw_entry(5),  // counters
                uniform_entry(6),     // params
            ],
        });
        let compact_atomic_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("compact_atomic_bg"),
            layout: &compact_atomic_bgl,
            entries: &[
                buf_entry(0, &bufs.insert_list),
                buf_entry(1, &bufs.uninserted),
                buf_entry(2, &bufs.vert_tet),
                buf_entry(3, &bufs.uninserted_temp),
                buf_entry(4, &bufs.vert_tet_temp),
                buf_entry(5, &bufs.counters),
                buf_entry(6, &compact_atomic_params),
            ],
        });
        let compact_atomic_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("compact_atomic_pl"),
            bind_group_layouts: &[&compact_atomic_bgl],
            immediate_size: 0,
        });
        let compact_atomic_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("compact_atomic"),
            layout: Some(&compact_atomic_pl),
            module: &compact_atomic_shader,
            entry_point: Some("compact_atomic"),
            compilation_options: Default::default(),
            cache: None,
        });

        // --- zero_compaction_flags (GPU buffer zeroing) ---
        let zero_compaction_flags_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("zero_compaction_flags.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/zero_compaction_flags.wgsl").into()),
        });
        let zero_compaction_flags_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("zero_compaction_flags_bgl"),
            entries: &[
                storage_rw_entry(0),  // flags
                uniform_entry(1),     // params
            ],
        });
        // Create a temporary params buffer for zero_compaction_flags (will share with compact_vertex_arrays)
        let zero_compact_params = GpuBuffers::create_params_buffer(device, [0, 0, 0, 0]);
        let zero_compaction_flags_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("zero_compaction_flags_bg"),
            layout: &zero_compaction_flags_bgl,
            entries: &[
                buf_entry(0, &bufs.compaction_flags),
                buf_entry(1, &zero_compact_params),
            ],
        });
        let zero_compaction_flags_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("zero_compaction_flags_pl"),
            bind_group_layouts: &[&zero_compaction_flags_bgl],
            immediate_size: 0,
        });
        let zero_compaction_flags_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("zero_compaction_flags"),
            layout: Some(&zero_compaction_flags_pl),
            module: &zero_compaction_flags_shader,
            entry_point: Some("zero_flags"),
            compilation_options: Default::default(),
            cache: None,
        });

        // --- compact_vertex_arrays (prefix-sum-based, 3 passes) ---
        let compact_vertex_arrays_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("compact_vertex_arrays_new.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/compact_vertex_arrays_new.wgsl").into()),
        });
        let compact_vertex_arrays_params = zero_compact_params;  // Reuse params buffer

        // Pass 1: mark_inserted
        let compact_mark_inserted_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("compact_mark_inserted_bgl"),
            entries: &[
                storage_ro_entry(2),  // insert_list
                storage_rw_entry(5),  // flags
                uniform_entry(8),     // params
            ],
        });
        let compact_mark_inserted_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("compact_mark_inserted_bg"),
            layout: &compact_mark_inserted_bgl,
            entries: &[
                buf_entry(2, &bufs.insert_list),
                buf_entry(5, &bufs.compaction_flags),
                buf_entry(8, &compact_vertex_arrays_params),
            ],
        });
        let compact_mark_inserted_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("compact_mark_inserted_pl"),
            bind_group_layouts: &[&compact_mark_inserted_bgl],
            immediate_size: 0,
        });
        let compact_mark_inserted_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("compact_mark_inserted"),
            layout: Some(&compact_mark_inserted_pl),
            module: &compact_vertex_arrays_shader,
            entry_point: Some("mark_inserted"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Pass 2: invert_flags
        let compact_invert_flags_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("compact_invert_flags_bgl"),
            entries: &[
                storage_rw_entry(5),  // flags
                uniform_entry(8),     // params
            ],
        });
        let compact_invert_flags_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("compact_invert_flags_bg"),
            layout: &compact_invert_flags_bgl,
            entries: &[
                buf_entry(5, &bufs.compaction_flags),
                buf_entry(8, &compact_vertex_arrays_params),
            ],
        });
        let compact_invert_flags_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("compact_invert_flags_pl"),
            bind_group_layouts: &[&compact_invert_flags_bgl],
            immediate_size: 0,
        });
        let compact_invert_flags_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("compact_invert_flags"),
            layout: Some(&compact_invert_flags_pl),
            module: &compact_vertex_arrays_shader,
            entry_point: Some("invert_flags"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Pass 3: scatter (uses prefix_sum_data as compact_map)
        let compact_scatter_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("compact_scatter_bgl"),
            entries: &[
                storage_ro_entry(0),  // uninserted_in
                storage_ro_entry(1),  // vert_tet_in
                storage_rw_entry(3),  // uninserted_out
                storage_rw_entry(4),  // vert_tet_out
                storage_rw_entry(5),  // flags (declared read_write in shader, even though only read here)
                storage_ro_entry(6),  // compact_map (prefix_sum_data)
                uniform_entry(8),     // params
            ],
        });
        let compact_scatter_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("compact_scatter_bg"),
            layout: &compact_scatter_bgl,
            entries: &[
                buf_entry(0, &bufs.uninserted),
                buf_entry(1, &bufs.vert_tet),
                buf_entry(3, &bufs.uninserted_temp),
                buf_entry(4, &bufs.vert_tet_temp),
                buf_entry(5, &bufs.compaction_flags),
                buf_entry(6, &bufs.prefix_sum_data),
                buf_entry(8, &compact_vertex_arrays_params),
            ],
        });
        let compact_scatter_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("compact_scatter_pl"),
            bind_group_layouts: &[&compact_scatter_bgl],
            immediate_size: 0,
        });
        let compact_scatter_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("compact_scatter"),
            layout: Some(&compact_scatter_pl),
            module: &compact_vertex_arrays_shader,
            entry_point: Some("scatter"),
            compilation_options: Default::default(),
            cache: None,
        });

        // --- pack_flags (for GPU prefix sum on compaction flags) ---
        let pack_flags_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pack_flags_to_vec4.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/pack_flags_to_vec4.wgsl").into()),
        });
        let pack_flags_params = GpuBuffers::create_params_buffer(device, [0, 0, 0, 0]);
        let pack_flags_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pack_flags_bgl"),
            entries: &[
                storage_ro_entry(0),  // flags (compaction_flags)
                storage_rw_entry(1),  // scan_in (scan_in_buf)
                uniform_entry(2),     // params
            ],
        });
        let pack_flags_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pack_flags_bg"),
            layout: &pack_flags_bgl,
            entries: &[
                buf_entry(0, &bufs.compaction_flags),
                buf_entry(1, &bufs.scan_in),
                buf_entry(2, &pack_flags_params),
            ],
        });
        let pack_flags_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pack_flags_pl"),
            bind_group_layouts: &[&pack_flags_bgl],
            immediate_size: 0,
        });
        let pack_flags_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pack_flags"),
            layout: Some(&pack_flags_pl),
            module: &pack_flags_shader,
            entry_point: Some("pack_flags_to_vec4"),
            compilation_options: Default::default(),
            cache: None,
        });

        // --- unpack_compact (for unpacking prefix sum result to compact_map) ---
        let unpack_compact_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("unpack_vec4_to_u32.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/unpack_vec4_to_u32.wgsl").into()),
        });
        let unpack_compact_params = GpuBuffers::create_params_buffer(device, [0, 0, 0, 0]);
        let unpack_compact_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("unpack_compact_bgl"),
            entries: &[
                storage_ro_entry(0),  // scan_out (scan_out_buf)
                storage_rw_entry(1),  // prefix_sum_data (used as compact_map)
                uniform_entry(2),     // params
            ],
        });
        let unpack_compact_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("unpack_compact_bg"),
            layout: &unpack_compact_bgl,
            entries: &[
                buf_entry(0, &bufs.scan_out),
                buf_entry(1, &bufs.prefix_sum_data),
                buf_entry(2, &unpack_compact_params),
            ],
        });
        let unpack_compact_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("unpack_compact_pl"),
            bind_group_layouts: &[&unpack_compact_bgl],
            immediate_size: 0,
        });
        let unpack_compact_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("unpack_compact"),
            layout: Some(&unpack_compact_pl),
            module: &unpack_compact_shader,
            entry_point: Some("unpack_vec4_to_u32"),
            compilation_options: Default::default(),
            cache: None,
        });

        // --- inclusive_to_exclusive (convert inclusive prefix sum → exclusive) ---
        let inclusive_to_exclusive_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("inclusive_to_exclusive.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/inclusive_to_exclusive.wgsl").into()),
        });
        let inclusive_to_exclusive_params = GpuBuffers::create_params_buffer(device, [0, 0, 0, 0]);
        let inclusive_to_exclusive_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("inclusive_to_exclusive_bgl"),
            entries: &[
                storage_ro_entry(0),  // flags
                storage_rw_entry(1),  // prefix_sum_data (in-place conversion)
                uniform_entry(2),     // params
            ],
        });
        let inclusive_to_exclusive_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("inclusive_to_exclusive_bg"),
            layout: &inclusive_to_exclusive_bgl,
            entries: &[
                buf_entry(0, &bufs.compaction_flags),
                buf_entry(1, &bufs.prefix_sum_data),
                buf_entry(2, &inclusive_to_exclusive_params),
            ],
        });
        let inclusive_to_exclusive_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("inclusive_to_exclusive_pl"),
            bind_group_layouts: &[&inclusive_to_exclusive_bgl],
            immediate_size: 0,
        });
        let inclusive_to_exclusive_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("inclusive_to_exclusive"),
            layout: Some(&inclusive_to_exclusive_pl),
            module: &inclusive_to_exclusive_shader,
            entry_point: Some("inclusive_to_exclusive"),
            compilation_options: Default::default(),
            cache: None,
        });

        // --- init_tet_to_flip (GPU buffer initialization) ---
        let init_tet_to_flip_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("init_tet_to_flip.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/init_tet_to_flip.wgsl").into()),
        });
        let init_tet_to_flip_params = GpuBuffers::create_params_buffer(device, [0, 0, 0, 0]);
        let init_tet_to_flip_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("init_tet_to_flip_bgl"),
            entries: &[
                storage_rw_entry(0),  // tet_to_flip
                uniform_entry(1),     // params
            ],
        });
        let init_tet_to_flip_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("init_tet_to_flip_bg"),
            layout: &init_tet_to_flip_bgl,
            entries: &[
                buf_entry(0, &bufs.tet_to_flip),
                buf_entry(1, &init_tet_to_flip_params),
            ],
        });
        let init_tet_to_flip_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("init_tet_to_flip_pl"),
            bind_group_layouts: &[&init_tet_to_flip_bgl],
            immediate_size: 0,
        });
        let init_tet_to_flip_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("init_tet_to_flip"),
            layout: Some(&init_tet_to_flip_pl),
            module: &init_tet_to_flip_shader,
            entry_point: Some("init_tet_to_flip"),
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
        let relocate_points_fast_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("relocate_points_fast_pl"),
            bind_group_layouts: &[&relocate_points_fast_bgl],
            immediate_size: 0,
        });
        let relocate_points_fast_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("relocate_points_fast"),
            layout: Some(&relocate_points_fast_pl),
            module: &relocate_points_fast_shader,
            entry_point: Some("relocate_points_fast"),
            compilation_options: Default::default(),
            cache: None,
        });

        // --- GPU Prefix Sum Pipelines ---

        // Load shaders
        let transform_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("transform_tet_alive.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/transform_tet_alive.wgsl").into()),
        });

        let prefix_sum_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("prefix_sum.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/prefix_sum.wgsl").into()),
        });

        let unpack_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("unpack_vec4_to_u32.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/unpack_vec4_to_u32.wgsl").into()),
        });

        // Params buffers
        let transform_params = GpuBuffers::create_params_buffer(device, [0, 0, 0, 0]);
        let unpack_params = GpuBuffers::create_params_buffer(device, [0, 0, 0, 0]);

        // 1. Transform pipeline
        let transform_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("transform_bgl"),
            entries: &[
                storage_ro_entry(0),  // tet_info
                storage_rw_entry(1),  // scan_in
                uniform_entry(2),     // params
            ],
        });

        let transform_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("transform_pl"),
            bind_group_layouts: &[&transform_bgl],
            immediate_size: 0,
        });

        let transform_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("transform_tet_alive"),
            layout: Some(&transform_pl),
            module: &transform_shader,
            entry_point: Some("transform_tet_alive"),
            compilation_options: Default::default(),
            cache: None,
        });

        let transform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("transform_bg"),
            layout: &transform_bgl,
            entries: &[
                buf_entry(0, &bufs.tet_info),
                buf_entry(1, &bufs.scan_in),
                buf_entry(2, &transform_params),
            ],
        });

        // 2. Reduce pipeline (prefix_sum.wgsl::reduce)
        let reduce_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("reduce_bgl"),
            entries: &[
                uniform_entry(0),     // info
                storage_rw_entry(1),  // scan_in
                storage_rw_entry(2),  // scan_out (unused in reduce)
                storage_rw_entry(3),  // scan_bump
                storage_rw_entry(4),  // reduction
                storage_rw_entry(5),  // misc
            ],
        });

        let reduce_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("reduce_pl"),
            bind_group_layouts: &[&reduce_bgl],
            immediate_size: 0,
        });

        let reduce_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("reduce"),
            layout: Some(&reduce_pl),
            module: &prefix_sum_shader,
            entry_point: Some("reduce"),
            compilation_options: Default::default(),
            cache: None,
        });

        let reduce_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("reduce_bg"),
            layout: &reduce_bgl,
            entries: &[
                buf_entry(0, &bufs.prefix_sum_info),
                buf_entry(1, &bufs.scan_in),
                buf_entry(2, &bufs.scan_out),
                buf_entry(3, &bufs.prefix_sum_bump),
                buf_entry(4, &bufs.reduction),
                buf_entry(5, &bufs.prefix_sum_misc),
            ],
        });

        // 3. Spine Scan pipeline (prefix_sum.wgsl::spine_scan)
        let spine_scan_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("spine_scan_bgl"),
            entries: &[
                uniform_entry(0),     // info
                storage_rw_entry(1),  // scan_in (unused in spine)
                storage_rw_entry(2),  // scan_out (unused in spine)
                storage_rw_entry(3),  // scan_bump
                storage_rw_entry(4),  // reduction
                storage_rw_entry(5),  // misc
            ],
        });

        let spine_scan_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("spine_scan_pl"),
            bind_group_layouts: &[&spine_scan_bgl],
            immediate_size: 0,
        });

        let spine_scan_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("spine_scan"),
            layout: Some(&spine_scan_pl),
            module: &prefix_sum_shader,
            entry_point: Some("spine_scan"),
            compilation_options: Default::default(),
            cache: None,
        });

        let spine_scan_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("spine_scan_bg"),
            layout: &spine_scan_bgl,
            entries: &[
                buf_entry(0, &bufs.prefix_sum_info),
                buf_entry(1, &bufs.scan_in),
                buf_entry(2, &bufs.scan_out),
                buf_entry(3, &bufs.prefix_sum_bump),
                buf_entry(4, &bufs.reduction),
                buf_entry(5, &bufs.prefix_sum_misc),
            ],
        });

        // 4. Downsweep pipeline (prefix_sum.wgsl::downsweep)
        let downsweep_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("downsweep_bgl"),
            entries: &[
                uniform_entry(0),     // info
                storage_rw_entry(1),  // scan_in
                storage_rw_entry(2),  // scan_out
                storage_rw_entry(3),  // scan_bump
                storage_rw_entry(4),  // reduction
                storage_rw_entry(5),  // misc
            ],
        });

        let downsweep_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("downsweep_pl"),
            bind_group_layouts: &[&downsweep_bgl],
            immediate_size: 0,
        });

        let downsweep_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("downsweep"),
            layout: Some(&downsweep_pl),
            module: &prefix_sum_shader,
            entry_point: Some("downsweep"),
            compilation_options: Default::default(),
            cache: None,
        });

        let downsweep_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("downsweep_bg"),
            layout: &downsweep_bgl,
            entries: &[
                buf_entry(0, &bufs.prefix_sum_info),
                buf_entry(1, &bufs.scan_in),
                buf_entry(2, &bufs.scan_out),
                buf_entry(3, &bufs.prefix_sum_bump),
                buf_entry(4, &bufs.reduction),
                buf_entry(5, &bufs.prefix_sum_misc),
            ],
        });

        // 5. Unpack pipeline
        let unpack_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("unpack_bgl"),
            entries: &[
                storage_ro_entry(0),  // scan_out
                storage_rw_entry(1),  // prefix_sum_data
                uniform_entry(2),     // params
            ],
        });

        let unpack_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("unpack_pl"),
            bind_group_layouts: &[&unpack_bgl],
            immediate_size: 0,
        });

        let unpack_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("unpack_vec4_to_u32"),
            layout: Some(&unpack_pl),
            module: &unpack_shader,
            entry_point: Some("unpack_vec4_to_u32"),
            compilation_options: Default::default(),
            cache: None,
        });

        let unpack_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("unpack_bg"),
            layout: &unpack_bgl,
            entries: &[
                buf_entry(0, &bufs.scan_out),
                buf_entry(1, &bufs.prefix_sum_data),
                buf_entry(2, &unpack_params),
            ],
        });


        Self {
            init_pipeline,
            init_params,
            init_vert_tet_pipeline,
            init_vert_tet_bind_group,
            vote_pipeline,
            vote_params,
            pick_pipeline,
            pick_bind_group,
            pick_params,
            build_insert_list_pipeline,
            build_insert_list_bind_group,
            build_insert_list_params,
            update_vert_free_pipeline,
            update_vert_free_bind_group,
            update_vert_free_params,
            split_points_pipeline,
            split_points_params,
            split_pipeline,
            split_params,
            update_uninserted_vert_tet_pipeline,
            update_uninserted_vert_tet_bind_group,
            update_uninserted_vert_tet_params,
            flip_vote_pipeline,
            flip_pipeline,
            flip_params,
            reset_votes_pipeline,
            reset_votes_bind_group,
            reset_votes_params,
            reset_flip_votes_pipeline,
            gather_pipeline,
            gather_params,
            check_delaunay_fast_pipeline,
            check_delaunay_fast_params,
            check_delaunay_exact_pipeline,
            check_delaunay_exact_params,
            allocate_flip23_slot_pipeline,
            allocate_flip23_slot_params,
            compact_if_negative_pipeline,
            compact_if_negative_bind_group,
            compact_if_negative_params,
            compact_atomic_pipeline,
            compact_atomic_bind_group,
            compact_atomic_params,
            zero_compaction_flags_pipeline,
            zero_compaction_flags_bind_group,
            compact_mark_inserted_pipeline,
            compact_mark_inserted_bind_group,
            compact_invert_flags_pipeline,
            compact_invert_flags_bind_group,
            compact_scatter_pipeline,
            compact_scatter_bind_group,
            compact_vertex_arrays_params,
            pack_flags_pipeline,
            pack_flags_bind_group,
            pack_flags_params,
            unpack_compact_pipeline,
            unpack_compact_bind_group,
            unpack_compact_params,
            inclusive_to_exclusive_pipeline,
            inclusive_to_exclusive_bind_group,
            inclusive_to_exclusive_params,
            init_tet_to_flip_pipeline,
            init_tet_to_flip_bind_group,
            init_tet_to_flip_params,
            relocate_points_fast_pipeline,
            relocate_points_fast_params,
            update_opp_pipeline,
            update_opp_params,
            donate_freed_tets_pipeline,
            donate_freed_tets_bgl,
            donate_freed_tets_params,
            update_flip_trace_pipeline,
            update_flip_trace_params,
            update_flip_trace_bgl,
            update_opp_bgl,
            relocate_points_fast_bgl,
            init_bgl,
            vote_bgl,
            split_points_bgl,
            split_bgl,
            flip_bgl,
            gather_bgl,
            compact_tets_bgl,
            mark_special_tets_bgl,
            check_delaunay_fast_bgl,
            check_delaunay_exact_bgl,
            mark_rejected_flips_bgl,
            allocate_flip23_slot_bgl,
            collect_free_slots_pipeline,
            collect_free_slots_bind_group,
            collect_free_slots_params,
            make_compact_map_pipeline,
            make_compact_map_bind_group,
            make_compact_map_params,
            compact_tets_pipeline,
            compact_tets_params,
            mark_special_tets_pipeline,
            mark_special_tets_params,
            repair_adjacency_pipeline,
            repair_adjacency_bgl,
            repair_adjacency_params,
            collect_active_tets_pipeline,
            collect_active_tets_bgl,
            collect_active_tets_params,
            mark_rejected_flips_pipeline,
            mark_rejected_flips_params,
            transform_pipeline,
            transform_bind_group,
            transform_params,
            reduce_pipeline,
            reduce_bind_group,
            spine_scan_pipeline,
            spine_scan_bind_group,
            downsweep_pipeline,
            downsweep_bind_group,
            unpack_pipeline,
            unpack_bind_group,
            unpack_params,
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
