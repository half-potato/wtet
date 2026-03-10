use super::GpuState;

/// Flip compaction mode (adaptive based on queue size)
/// CUDA Reference: GpuDelaunay.cu:703, 950-958
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlipCompactMode {
    /// Small queues: Use atomic collection in mark_rejected_flips, skip separate compaction
    /// Used when: flip_queue_size < 256
    CollectCompact,

    /// Large queues: Mark rejected flips, then run 2-pass compaction
    /// Used when: flip_queue_size >= 256
    MarkCompact,
}

impl FlipCompactMode {
    /// Select compaction mode based on queue size
    /// CUDA: actNum < BlocksPerGrid * ThreadsPerBlock (= 256)
    pub fn select(flip_queue_size: u32) -> Self {
        if flip_queue_size < 256 {
            Self::CollectCompact
        } else {
            Self::MarkCompact
        }
    }
}

/// Dispatch helpers that encode and submit compute passes.
impl GpuState {
    /// Dispatch the init kernel (creates 5-tet super-tetrahedron).
    /// OPTIMIZATION: Now split into two passes:
    /// 1. make_first_tetra - Single thread creates 5 tets
    /// 2. init_vert_tet - Parallel threads initialize vert_tet array
    pub fn dispatch_init(&self, encoder: &mut wgpu::CommandEncoder) {
        // PASS 1: Create 5-tet super-tetrahedron (single thread)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("init_make_first_tetra"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.init_pipeline);

            // Create dynamic bind group
            // For large datasets, bind only the 5 tets needed for super-tet initialization
            let (tets_size, tet_opp_size) = if self.use_partial_binding {
                let size = 5 * 16; // 5 tets × 16 bytes each
                (Some(wgpu::BufferSize::new(size).unwrap()), Some(wgpu::BufferSize::new(size).unwrap()))
            } else {
                (None, None)
            };

            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("init_bg_dynamic"),
                layout: &self.pipelines.init_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.buffers.points.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &self.buffers.tets,
                            offset: 0,
                            size: tets_size,
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &self.buffers.tet_opp,
                            offset: 0,
                            size: tet_opp_size,
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.buffers.tet_info.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.buffers.vert_tet.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: self.buffers.free_arr.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: self.buffers.vert_free_arr.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: self.buffers.counters.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: self.pipelines.init_params.as_entire_binding(),
                    },
                ],
            });
            pass.set_bind_group(0, Some(&bind_group), &[]);

            pass.dispatch_workgroups(1, 1, 1);
        }

        // PASS 2: Initialize vert_tet array in parallel
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("init_vert_tet"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.init_vert_tet_pipeline);
            pass.set_bind_group(0, &self.pipelines.init_vert_tet_bind_group, &[]);

            let num_points = self.num_points;
            let workgroups = (num_points + 255) / 256;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
    }

    /// Dispatch vote for point.
    pub fn dispatch_vote(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        num_uninserted: u32,
    ) {
        // Pass num_uninserted and inf_idx to shader
        // inf_idx is at position num_points+4 (after 4 super-tet vertices)
        // Vertex layout: [0..N-1] real points, [N..N+3] super-tet, [N+4] infinity
        let inf_idx = self.num_points + 4;
        queue.write_buffer(
            &self.pipelines.vote_params,
            0,
            bytemuck::cast_slice(&[num_uninserted, inf_idx, 0u32, 0u32]),
        );

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("vote"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipelines.vote_pipeline);

        // Create dynamic bind group (partial binding for large datasets, full binding for small)
        let active_tets = self.current_tet_num;
        let tets_size = if self.use_partial_binding {
            Some(wgpu::BufferSize::new((active_tets as u64) * 16).unwrap())
        } else {
            None // Full binding for small datasets
        };

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vote_bg_dynamic"),
            layout: &self.pipelines.vote_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.buffers.points.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.buffers.tets,
                        offset: 0,
                        size: tets_size,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.buffers.tet_info.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.buffers.tet_vote.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.buffers.vert_sphere.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.buffers.vert_tet.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.buffers.uninserted.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: self.pipelines.vote_params.as_entire_binding(),
                },
            ],
        });
        pass.set_bind_group(0, Some(&bind_group), &[]);

        pass.dispatch_workgroups(div_ceil(num_uninserted, 256), 1, 1);
    }

    /// Dispatch pick winner.
    /// Uses VERTEX-PARALLEL iteration matching CUDA's kerPickWinnerPoint (KerDivision.cu:311-335).
    /// Each uninserted vertex checks if it won its tet and competes via atomicMin.
    /// Entry point: vote.wgsl::pick_winner_point
    pub fn dispatch_pick_winner(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        num_uninserted: u32,
    ) {
        queue.write_buffer(
            &self.pipelines.pick_params,
            0,
            bytemuck::cast_slice(&[num_uninserted, 0u32, 0u32, 0u32]),
        );

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("pick_winner"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipelines.pick_pipeline);
        pass.set_bind_group(0, Some(&self.pipelines.pick_bind_group), &[]);
        pass.dispatch_workgroups(div_ceil(num_uninserted, 256), 1, 1);
    }

    /// Dispatch build insert list (filters exact winners after pick_winner).
    /// Second pass of CUDA's two-pass winner selection to prevent duplicate insertions.
    pub fn dispatch_build_insert_list(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        num_uninserted: u32,
    ) {
        queue.write_buffer(
            &self.pipelines.build_insert_list_params,
            0,
            bytemuck::cast_slice(&[num_uninserted, 0u32, 0u32, 0u32]),
        );

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("build_insert_list"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipelines.build_insert_list_pipeline);
        pass.set_bind_group(0, Some(&self.pipelines.build_insert_list_bind_group), &[]);
        pass.dispatch_workgroups(div_ceil(num_uninserted, 256), 1, 1);
    }

    // REMOVED: dispatch_mark_split
    // mark_split.wgsl was obsolete - pick_winner_point already populates tet_to_vert
    // See MEMORY.md "Obsolete Shader Cleanup (2026-03-04)"

    /// Dispatch split_points (updates vert_tet for uninserted vertices whose tets are splitting).
    pub fn dispatch_split_points(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        num_uninserted: u32,
    ) {
        // Write params with actual count after compaction (not buffer size!)
        queue.write_buffer(
            &self.pipelines.split_points_params,
            0,
            bytemuck::cast_slice(&[num_uninserted, 0u32, 0u32, 0u32]),
        );

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("split_points"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipelines.split_points_pipeline);

        // Create dynamic bind group
        // CRITICAL: Cap active_tets to max_tets (buffer capacity)
        let active_tets = self.current_tet_num.min(self.max_tets);
        let (tets_size, tet_opp_size) = if self.use_partial_binding {
            let size = (active_tets as u64) * 16;
            let max_buffer_size = (self.max_tets as u64) * 16;
            // Cap to actual buffer size
            let capped_size = size.min(max_buffer_size);
            (Some(wgpu::BufferSize::new(capped_size).unwrap()), Some(wgpu::BufferSize::new(capped_size).unwrap()))
        } else {
            (None, None)
        };

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("split_points_bg_dynamic"),
            layout: &self.pipelines.split_points_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.buffers.points.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.buffers.tets,
                        offset: 0,
                        size: tets_size,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.buffers.tet_opp,
                        offset: 0,
                        size: tet_opp_size,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.buffers.tet_info.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.buffers.vert_tet.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.buffers.free_arr.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.buffers.vert_free_arr.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: self.buffers.counters.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: self.buffers.uninserted.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: self.buffers.tet_to_vert.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: self.pipelines.split_points_params.as_entire_binding(),
                },
            ],
        });
        pass.set_bind_group(0, Some(&bind_group), &[]);

        pass.dispatch_workgroups(div_ceil(num_uninserted, 256), 1, 1);
    }

    /// Dispatch split tetra.
    pub fn dispatch_split(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        num_insertions: u32,
    ) {
        // params: x = num_insertions, y = inf_idx, z = current_tet_num
        // inf_idx is at position num_points+4 (AFTER the 4 super-tet vertices)
        // Super-tet vertices are at [num_points, num_points+1, num_points+2, num_points+3]
        let inf_idx = self.num_points + 4;
        let current_tet_num = self.current_tet_num;

        let params_data = [num_insertions, inf_idx, current_tet_num, 0u32];
        eprintln!("[DISPATCH_SPLIT] Writing params: {:?}", params_data);
        queue.write_buffer(
            &self.pipelines.split_params,
            0,
            bytemuck::cast_slice(&params_data),
        );

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("split"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipelines.split_pipeline);

        // Create dynamic bind group
        let active_tets = current_tet_num;
        let (tets_size, tet_opp_size) = if self.use_partial_binding {
            let size = (active_tets as u64) * 16;
            (Some(wgpu::BufferSize::new(size).unwrap()), Some(wgpu::BufferSize::new(size).unwrap()))
        } else {
            (None, None)
        };

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("split_bg_dynamic"),
            layout: &self.pipelines.split_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.buffers.tets,
                        offset: 0,
                        size: tets_size,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.buffers.tet_opp,
                        offset: 0,
                        size: tet_opp_size,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.buffers.tet_info.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.buffers.vert_tet.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.buffers.insert_list.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.buffers.free_arr.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.buffers.vert_free_arr.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: self.buffers.counters.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: self.buffers.flip_queue.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: self.buffers.tet_to_vert.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: self.pipelines.split_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: self.buffers.block_owner.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: self.buffers.uninserted.as_entire_binding(),
                },
            ],
        });
        pass.set_bind_group(0, Some(&bind_group), &[]);

        pass.dispatch_workgroups(div_ceil(num_insertions, 256), 1, 1);
    }

    /// Dispatch update_uninserted_vert_tet to fix dead tet references.
    pub fn dispatch_update_uninserted_vert_tet(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        num_uninserted: u32,
    ) {
        log::debug!(
            "Updating vert_tet for {} uninserted vertices (max_tets={})",
            num_uninserted,
            self.max_tets
        );

        queue.write_buffer(
            &self.pipelines.update_uninserted_vert_tet_params,
            0,
            bytemuck::cast_slice(&[num_uninserted, self.max_tets, 0u32, 0u32]),
        );

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("update_uninserted_vert_tet"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipelines.update_uninserted_vert_tet_pipeline);
        pass.set_bind_group(0, Some(&self.pipelines.update_uninserted_vert_tet_bind_group), &[]);
        pass.dispatch_workgroups(div_ceil(num_uninserted, 256), 1, 1);
    }

    /// Dispatch flip check.
    ///
    /// When `use_alternate` is true, uses the swapped bind group where
    /// flip_queue_next is read and flip_queue is written.
    pub fn dispatch_flip(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        queue_size: u32,
        inf_idx: u32,
        use_alternate: bool,
    ) {
        queue.write_buffer(
            &self.pipelines.flip_params,
            0,
            bytemuck::cast_slice(&[queue_size, inf_idx, 0u32, 0u32]),
        );

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("flip"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipelines.flip_pipeline);

        // Select queues based on use_alternate (swap read/write buffers)
        let (flip_queue_read, flip_queue_write) = if use_alternate {
            (&self.buffers.flip_queue_next, &self.buffers.flip_queue)
        } else {
            (&self.buffers.flip_queue, &self.buffers.flip_queue_next)
        };

        // Create dynamic bind group
        // CRITICAL: Cap active_tets to max_tets (buffer capacity)
        let active_tets = self.current_tet_num.min(self.max_tets);
        let (tets_size, tet_opp_size) = if self.use_partial_binding {
            let size = (active_tets as u64) * 16;
            let max_buffer_size = (self.max_tets as u64) * 16;
            // Cap to actual buffer size
            let capped_size = size.min(max_buffer_size);
            (Some(wgpu::BufferSize::new(capped_size).unwrap()), Some(wgpu::BufferSize::new(capped_size).unwrap()))
        } else {
            (None, None)
        };

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("flip_bg_dynamic"),
            layout: &self.pipelines.flip_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.buffers.points.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.buffers.tets,
                        offset: 0,
                        size: tets_size,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.buffers.tet_opp,
                        offset: 0,
                        size: tet_opp_size,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.buffers.tet_info.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.buffers.free_arr.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.buffers.vert_free_arr.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.buffers.counters.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: flip_queue_read.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: flip_queue_write.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: self.buffers.flip_count.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: self.pipelines.flip_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: self.buffers.block_owner.as_entire_binding(),
                },
            ],
        });
        pass.set_bind_group(0, Some(&bind_group), &[]);

        pass.dispatch_workgroups(div_ceil(queue_size, 256), 1, 1);
    }

    /// Dispatch reset votes.
    /// CRITICAL: Updates params buffer with current num_uninserted before dispatch.
    /// CUDA Reference: GpuDelaunay.cu:953 (tetSphereVec.assign with current vertNum)
    pub fn dispatch_reset_votes(&self, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue, num_uninserted: u32) {
        // Update params buffer with current num_uninserted (changes each iteration)
        // CUDA allocates fresh arrays each iteration; WGPU reuses buffers but updates size params
        queue.write_buffer(
            &self.pipelines.reset_votes_params,
            0,
            bytemuck::cast_slice(&[self.max_tets, num_uninserted, 0u32, 0u32]),
        );

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("reset_votes"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipelines.reset_votes_pipeline);
        pass.set_bind_group(0, Some(&self.pipelines.reset_votes_bind_group), &[]);
        pass.dispatch_workgroups(div_ceil(num_uninserted, 256), 1, 1);
    }

    /// Dispatch gather failed vertices.
    pub fn dispatch_gather(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gather"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipelines.gather_pipeline);

        // Create dynamic bind group
        // CRITICAL: Cap active_tets to max_tets (buffer capacity)
        let active_tets = self.current_tet_num.min(self.max_tets);
        let (tets_size, tet_opp_size) = if self.use_partial_binding {
            let size = (active_tets as u64) * 16;
            let max_buffer_size = (self.max_tets as u64) * 16;
            // Cap to actual buffer size
            let capped_size = size.min(max_buffer_size);
            (Some(wgpu::BufferSize::new(capped_size).unwrap()), Some(wgpu::BufferSize::new(capped_size).unwrap()))
        } else {
            (None, None)
        };

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gather_bg_dynamic"),
            layout: &self.pipelines.gather_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.buffers.points.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.buffers.tets,
                        offset: 0,
                        size: tets_size,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.buffers.tet_opp,
                        offset: 0,
                        size: tet_opp_size,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.buffers.tet_info.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.buffers.failed_verts.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.buffers.counters.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.pipelines.gather_params.as_entire_binding(),
                },
            ],
        });
        pass.set_bind_group(0, Some(&bind_group), &[]);

        pass.dispatch_workgroups(div_ceil(self.max_tets, 256), 1, 1);
    }

    /// Reset the inserted_count counter to 0.
    pub fn reset_inserted_counter(&self, queue: &wgpu::Queue) {
        // Counter layout: [free_count, active_count, inserted_count, failed_count, ...]
        // inserted_count is at offset 8
        queue.write_buffer(
            &self.buffers.counters,
            8,
            bytemuck::cast_slice(&[0u32]),
        );
    }

    /// Reset flip_count to 0.
    pub fn reset_flip_count(&self, queue: &wgpu::Queue) {
        queue.write_buffer(
            &self.buffers.flip_count,
            0,
            bytemuck::cast_slice(&[0u32]),
        );
    }

    /// Dispatch collect_free_slots (compactTetras step 1: collect dead tet indices).
    pub fn dispatch_collect_free_slots(&self, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue, new_tet_num: u32) {
        queue.write_buffer(
            &self.pipelines.collect_free_slots_params,
            0,
            bytemuck::cast_slice(&[new_tet_num, 0u32, 0u32, 0u32]),
        );

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("collect_free_slots"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipelines.collect_free_slots_pipeline);
        pass.set_bind_group(0, Some(&self.pipelines.collect_free_slots_bind_group), &[]);
        pass.dispatch_workgroups(div_ceil(new_tet_num, 256), 1, 1);
    }

    /// Dispatch make_compact_map (compactTetras step 2: create old→new mapping).
    pub fn dispatch_make_compact_map(&self, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue, new_tet_num: u32, total_tet_num: u32) {
        queue.write_buffer(
            &self.pipelines.make_compact_map_params,
            0,
            bytemuck::cast_slice(&[new_tet_num, total_tet_num, 0u32, 0u32]),
        );

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("make_compact_map"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipelines.make_compact_map_pipeline);
        pass.set_bind_group(0, Some(&self.pipelines.make_compact_map_bind_group), &[]);

        let count = if total_tet_num > new_tet_num {
            total_tet_num - new_tet_num
        } else {
            0
        };
        pass.dispatch_workgroups(div_ceil(count, 256), 1, 1);
    }

    /// Dispatch compact_tets (compactTetras step 3: physically compact alive tets).
    pub fn dispatch_compact_tets(&self, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue, new_tet_num: u32, total_tet_num: u32) {
        queue.write_buffer(
            &self.pipelines.compact_tets_params,
            0,
            bytemuck::cast_slice(&[new_tet_num, total_tet_num, 0u32, 0u32]),
        );

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("compact_tets"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipelines.compact_tets_pipeline);

        // Create dynamic bind group
        let (tets_size, tet_opp_size) = if self.use_partial_binding {
            let size = (total_tet_num as u64) * 16;
            (Some(wgpu::BufferSize::new(size).unwrap()), Some(wgpu::BufferSize::new(size).unwrap()))
        } else {
            (None, None)
        };

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("compact_tets_bg_dynamic"),
            layout: &self.pipelines.compact_tets_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.buffers.tet_info.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.buffers.prefix_sum_data.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.buffers.tets,
                        offset: 0,
                        size: tets_size,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.buffers.tet_opp,
                        offset: 0,
                        size: tet_opp_size,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.pipelines.compact_tets_params.as_entire_binding(),
                },
            ],
        });
        pass.set_bind_group(0, Some(&bind_group), &[]);

        let count = if total_tet_num > new_tet_num {
            total_tet_num - new_tet_num
        } else {
            0
        };
        pass.dispatch_workgroups(div_ceil(count, 256), 1, 1);
    }

    /// Dispatch mark_special_tets (clears OPP_SPECIAL flags between fast/exact flipping).
    pub fn dispatch_mark_special_tets(&self, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue, tet_num: u32) {
        queue.write_buffer(
            &self.pipelines.mark_special_tets_params,
            0,
            bytemuck::cast_slice(&[tet_num, 0u32, 0u32, 0u32]),
        );

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("mark_special_tets"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipelines.mark_special_tets_pipeline);

        // Create dynamic bind group
        let tet_opp_size = if self.use_partial_binding {
            Some(wgpu::BufferSize::new((tet_num as u64) * 16).unwrap())
        } else {
            None
        };

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mark_special_tets_bg_dynamic"),
            layout: &self.pipelines.mark_special_tets_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.buffers.tet_info.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.buffers.tet_opp,
                        offset: 0,
                        size: tet_opp_size,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.buffers.act_tet_vec.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.buffers.counters.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.pipelines.mark_special_tets_params.as_entire_binding(),
                },
            ],
        });
        pass.set_bind_group(0, Some(&bind_group), &[]);

        pass.dispatch_workgroups(div_ceil(tet_num, 256), 1, 1);
    }

    /// Dispatch update_flip_trace to build flip history chains.
    /// Port of kerUpdateFlipTrace from KerDivision.cu:742-782
    pub fn dispatch_update_flip_trace(&self, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue, org_flip_num: u32, flip_num: u32) {
        queue.write_buffer(
            &self.pipelines.update_flip_trace_params,
            0,
            bytemuck::cast_slice(&[org_flip_num, flip_num, 0u32, 0u32]),
        );

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("update_flip_trace"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipelines.update_flip_trace_pipeline);

        // Create dynamic bind group
        // CRITICAL: Cap binding size to actual buffer capacity (max_flips = max_tets/2)
        let (offset, size) = if self.use_partial_binding {
            let offset = (org_flip_num as u64) * 32; // FlipItem = 32 bytes
            let requested_size = (flip_num as u64) * 32;
            let max_flip_arr_size = (self.max_flips as u64) * 32;
            // Cap size to remaining buffer space after offset
            let capped_size = requested_size.min(max_flip_arr_size.saturating_sub(offset));
            (offset, Some(wgpu::BufferSize::new(capped_size).unwrap()))
        } else {
            (0, None)
        };

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("update_flip_trace_bg_dynamic"),
            layout: &self.pipelines.update_flip_trace_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.buffers.flip_arr,
                        offset,
                        size,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.buffers.tet_to_flip.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.pipelines.update_flip_trace_params.as_entire_binding(),
                },
            ],
        });
        pass.set_bind_group(0, Some(&bind_group), &[]);

        pass.dispatch_workgroups(div_ceil(flip_num, 256), 1, 1);
    }

    pub fn dispatch_update_block_vert_free_list(&self, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue, ins_num: u32, old_ins_num: u32) {
        // ...stub...
    }

    pub fn dispatch_update_block_opp_tet_idx(&self, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue, old_inf_block_idx: u32, new_inf_block_idx: u32, old_tet_num: u32) {
        // ...stub...
    }

    pub fn dispatch_shift_inf_free_idx(&self, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue, inf_idx: u32, start: u32, shift: u32) {
        // ...stub...
    }

    pub fn dispatch_update_tet_idx(&self, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue, old_inf_block_idx: u32, new_inf_block_idx: u32, vec_size: u32) {
        // ...stub...
    }

    pub fn dispatch_shift_opp_tet_idx(&self, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue, tet_num: u32, start: u32, shift: u32) {
        // ...stub...
    }

    pub fn dispatch_shift_tet_idx(&self, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue, vec_size: u32, start: u32, shift: u32) {
        // ...stub...
    }

    pub fn dispatch_make_reverse_map(&self, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue, ins_vert_num: u32, num: u32) {
        // ...stub...
    }

    /// Dispatch update opp (CRITICAL - flip adjacency updates).
    pub fn dispatch_update_opp(&self, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue, org_flip_num: u32, flip_num: u32) {
        queue.write_buffer(&self.pipelines.update_opp_params, 0, bytemuck::cast_slice(&[org_flip_num, flip_num, 0u32, 0u32]));
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("update_opp"), timestamp_writes: None });
        pass.set_pipeline(&self.pipelines.update_opp_pipeline);

        // Create dynamic bind group
        // CRITICAL: Cap binding size to actual buffer capacity (max_flips = max_tets/2)
        let (offset, size) = if self.use_partial_binding {
            let offset = (org_flip_num as u64) * 32; // FlipItem = 32 bytes
            let requested_size = (flip_num as u64) * 32;
            let max_flip_arr_size = (self.max_flips as u64) * 32;
            // Cap size to remaining buffer space after offset
            let capped_size = requested_size.min(max_flip_arr_size.saturating_sub(offset));
            (offset, Some(wgpu::BufferSize::new(capped_size).unwrap()))
        } else {
            (0, None)
        };

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("update_opp_bg_dynamic"),
            layout: &self.pipelines.update_opp_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.buffers.flip_arr,
                        offset,
                        size,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.buffers.tet_opp.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.buffers.tet_msg_arr.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.buffers.encoded_face_vi_arr.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.pipelines.update_opp_params.as_entire_binding(),
                },
            ],
        });
        pass.set_bind_group(0, Some(&bind_group), &[]);

        pass.dispatch_workgroups(div_ceil(flip_num, 256), 1, 1);
    }

    /// Dispatch mark rejected flips (flip validation).
    pub fn dispatch_mark_rejected_flips(&self, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue, act_tet_num: u32, vote_offset: i32, compact_mode: bool) {
        queue.write_buffer(&self.pipelines.mark_rejected_flips_params, 0, bytemuck::cast_slice(&[act_tet_num, vote_offset as u32, if compact_mode { 1u32 } else { 0u32 }, 0u32]));
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("mark_rejected_flips"), timestamp_writes: None });
        pass.set_pipeline(&self.pipelines.mark_rejected_flips_pipeline);

        // Create dynamic bind group
        let tet_opp_size = if self.use_partial_binding {
            Some(wgpu::BufferSize::new((act_tet_num as u64) * 16).unwrap())
        } else {
            None
        };

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mark_rejected_flips_bg_dynamic"),
            layout: &self.pipelines.mark_rejected_flips_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.buffers.act_tet_vec.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.buffers.tet_opp,
                        offset: 0,
                        size: tet_opp_size,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.buffers.tet_vote.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.buffers.tet_info.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.buffers.vote_arr.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.buffers.flip_to_tet.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.buffers.counters.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: self.pipelines.mark_rejected_flips_params.as_entire_binding(),
                },
            ],
        });
        pass.set_bind_group(0, Some(&bind_group), &[]);

        pass.dispatch_workgroups(div_ceil(act_tet_num, 256), 1, 1);
    }

    /// Dispatch check delaunay fast (flip voting).
    pub fn dispatch_check_delaunay_fast(&self, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue, act_tet_num: u32, vote_offset: u32) {
        let inf_idx = self.num_points + 4; // Infinity vertex (after 4 super-tet vertices)
        queue.write_buffer(&self.pipelines.check_delaunay_fast_params, 0, bytemuck::cast_slice(&[act_tet_num, vote_offset, inf_idx, 0u32]));
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("check_delaunay_fast"), timestamp_writes: None });
        pass.set_pipeline(&self.pipelines.check_delaunay_fast_pipeline);

        // Create dynamic bind group
        let (tets_size, tet_opp_size) = if self.use_partial_binding {
            let size = (act_tet_num as u64) * 16;
            (Some(wgpu::BufferSize::new(size).unwrap()), Some(wgpu::BufferSize::new(size).unwrap()))
        } else {
            (None, None)
        };

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("check_delaunay_fast_bg_dynamic"),
            layout: &self.pipelines.check_delaunay_fast_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.buffers.act_tet_vec.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.buffers.tets,
                        offset: 0,
                        size: tets_size,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.buffers.tet_opp,
                        offset: 0,
                        size: tet_opp_size,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.buffers.tet_info.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.buffers.tet_vote.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.buffers.vote_arr.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.buffers.counters.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: self.buffers.points.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: self.pipelines.check_delaunay_fast_params.as_entire_binding(),
                },
            ],
        });
        pass.set_bind_group(0, Some(&bind_group), &[]);

        pass.dispatch_workgroups(div_ceil(act_tet_num, 256), 1, 1);
    }

    /// Dispatch check delaunay exact (with DD + SoS predicates).
    pub fn dispatch_check_delaunay_exact(&self, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue, act_tet_num: u32, vote_offset: u32) {
        let inf_idx = self.num_points + 4; // Infinity vertex (after 4 super-tet vertices)
        queue.write_buffer(&self.pipelines.check_delaunay_exact_params, 0, bytemuck::cast_slice(&[act_tet_num, vote_offset, inf_idx, 0u32]));
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("check_delaunay_exact"), timestamp_writes: None });
        pass.set_pipeline(&self.pipelines.check_delaunay_exact_pipeline);

        // Create dynamic bind group
        let (tets_size, tet_opp_size) = if self.use_partial_binding {
            let size = (act_tet_num as u64) * 16;
            (Some(wgpu::BufferSize::new(size).unwrap()), Some(wgpu::BufferSize::new(size).unwrap()))
        } else {
            (None, None)
        };

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("check_delaunay_exact_bg_dynamic"),
            layout: &self.pipelines.check_delaunay_exact_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.buffers.act_tet_vec.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.buffers.tets,
                        offset: 0,
                        size: tets_size,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.buffers.tet_opp,
                        offset: 0,
                        size: tet_opp_size,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.buffers.tet_info.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.buffers.tet_vote.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.buffers.vote_arr.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.buffers.counters.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: self.buffers.points.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: self.pipelines.check_delaunay_exact_params.as_entire_binding(),
                },
            ],
        });
        pass.set_bind_group(0, Some(&bind_group), &[]);

        pass.dispatch_workgroups(div_ceil(act_tet_num, 256), 1, 1);
    }

    /// Dispatch allocate flip23 slot.
    pub fn dispatch_allocate_flip23_slot(&self, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue, flip_num: u32, inf_idx: u32, tet_num: u32) {
        queue.write_buffer(&self.pipelines.allocate_flip23_slot_params, 0, bytemuck::cast_slice(&[flip_num, inf_idx, tet_num, 0u32]));
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("allocate_flip23_slot"), timestamp_writes: None });
        pass.set_pipeline(&self.pipelines.allocate_flip23_slot_pipeline);

        // Create dynamic bind group
        let tets_size = if self.use_partial_binding {
            Some(wgpu::BufferSize::new((tet_num as u64) * 16).unwrap())
        } else {
            None
        };

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("allocate_flip23_slot_bg_dynamic"),
            layout: &self.pipelines.allocate_flip23_slot_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.buffers.flip_to_tet.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.buffers.tets,
                        offset: 0,
                        size: tets_size,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.buffers.vert_free_arr.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.buffers.free_arr.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.buffers.flip23_new_slot.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.pipelines.allocate_flip23_slot_params.as_entire_binding(),
                },
            ],
        });
        pass.set_bind_group(0, Some(&bind_group), &[]);

        pass.dispatch_workgroups(div_ceil(flip_num, 256), 1, 1);
    }

    /// Dispatch compact if negative (two-pass compaction).
    /// Returns the compacted count via reading counter[0] after both passes.
    pub fn dispatch_compact_if_negative(&self, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue, input_size: u32) {
        // CRITICAL: Reset counters[0] and counters[1] to zero before atomic operations!
        queue.write_buffer(&self.buffers.counters, 0, bytemuck::cast_slice(&[0u32, 0u32]));

        // Pass 1: Count non-negative values
        queue.write_buffer(&self.pipelines.compact_if_negative_params, 0, bytemuck::cast_slice(&[input_size, 0u32, 0u32, 0u32]));
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("compact_if_negative_pass1"), timestamp_writes: None });
        pass.set_pipeline(&self.pipelines.compact_if_negative_pipeline);
        pass.set_bind_group(0, Some(&self.pipelines.compact_if_negative_bind_group), &[]);
        pass.dispatch_workgroups(div_ceil(input_size, 256), 1, 1);
        drop(pass);

        // Pass 2: Compact non-negative values
        queue.write_buffer(&self.pipelines.compact_if_negative_params, 0, bytemuck::cast_slice(&[input_size, 1u32, 0u32, 0u32]));
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("compact_if_negative_pass2"), timestamp_writes: None });
        pass.set_pipeline(&self.pipelines.compact_if_negative_pipeline);
        pass.set_bind_group(0, Some(&self.pipelines.compact_if_negative_bind_group), &[]);
        pass.dispatch_workgroups(div_ceil(input_size, 256), 1, 1);
    }

    /// Compact vertex arrays using atomic counters (fast path for small datasets).
    /// Two-pass algorithm: Pass 0 counts non-inserted, Pass 1 scatters them.
    /// Much faster than prefix sum for typical datasets (< 100k elements).
    /// Returns a new encoder and the new uninserted count.
    pub async fn dispatch_compact_vertex_arrays_atomic(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        num_uninserted: u32,
        num_inserted: u32,
    ) -> (wgpu::CommandEncoder, u32) {
        // Reset first 2 counters to 0 (used for atomic counting)
        // counters buffer is 8 u32s total, only reset first 2
        queue.write_buffer(
            &self.buffers.counters,
            0,
            bytemuck::cast_slice(&[0u32, 0u32]),
        );

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("compact_atomic"),
        });

        // Pass 0: Count non-inserted positions
        queue.write_buffer(
            &self.pipelines.compact_atomic_params,
            0,
            bytemuck::cast_slice(&[num_uninserted, num_inserted, 0u32, 0u32]),
        );
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("compact_atomic_count"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.compact_atomic_pipeline);
            pass.set_bind_group(0, Some(&self.pipelines.compact_atomic_bind_group), &[]);
            pass.dispatch_workgroups(div_ceil(num_uninserted, 256), 1, 1);
        }

        // Pass 1: Scatter non-inserted vertices
        queue.write_buffer(
            &self.pipelines.compact_atomic_params,
            0,
            bytemuck::cast_slice(&[num_uninserted, num_inserted, 1u32, 0u32]),
        );
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("compact_atomic_scatter"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.compact_atomic_pipeline);
            pass.set_bind_group(0, Some(&self.pipelines.compact_atomic_bind_group), &[]);
            pass.dispatch_workgroups(div_ceil(num_uninserted, 256), 1, 1);
        }

        // Submit and read count (single readback of 4 bytes)
        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });

        let new_count = self.buffers.read_collection_count(device, queue).await;

        // Create encoder for buffer copies (caller will copy uninserted_copy → uninserted)
        let encoder2 = device.create_command_encoder(&Default::default());
        (encoder2, new_count)
    }

    /// Dispatch compact vertex arrays (two-pass compaction).
    /// Compacts both uninserted and vert_tet arrays by removing inserted vertices.
    ///
    /// CRITICAL: This function submits the encoder internally between passes to ensure
    /// params buffer writes are properly synchronized. Returns a new encoder for the caller.
    pub async fn dispatch_compact_vertex_arrays(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        num_uninserted: u32,
        num_inserted: u32,
    ) -> (wgpu::CommandEncoder, u32) {
        // New algorithm: mark → invert → CPU_prefix_sum → scatter
        // Eliminates O(N²) linear search!
        // CPU prefix sum is O(N) which is MUCH faster than O(N×M) linear search
        // (GPU prefix sum integration can be added later for further optimization)

        // Zero-initialize compaction_flags buffer
        let zero_flags = vec![0u32; num_uninserted as usize];
        queue.write_buffer(
            &self.buffers.compaction_flags,
            0,
            bytemuck::cast_slice(&zero_flags),
        );

        // Update params for all passes
        queue.write_buffer(
            &self.pipelines.compact_vertex_arrays_params,
            0,
            bytemuck::cast_slice(&[num_uninserted, num_inserted, 0u32, 0u32]),
        );

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("compact_vertex_arrays"),
        });

        // Pass 1: Mark inserted positions
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("compact_mark_inserted"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.compact_mark_inserted_pipeline);
            pass.set_bind_group(0, Some(&self.pipelines.compact_mark_inserted_bind_group), &[]);
            pass.dispatch_workgroups(div_ceil(num_inserted, 256), 1, 1);
        }

        // Pass 2: Invert flags (inserted→0, not_inserted→1)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("compact_invert_flags"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.compact_invert_flags_pipeline);
            pass.set_bind_group(0, Some(&self.pipelines.compact_invert_flags_bind_group), &[]);
            pass.dispatch_workgroups(div_ceil(num_uninserted, 256), 1, 1);
        }

        // Submit passes 1-2 before GPU prefix sum
        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });

        // Pass 3: GPU prefix sum on flags → compact_map
        // This is the key optimization: runs entirely on GPU, no CPU readback!
        let new_count = self.dispatch_gpu_prefix_sum_on_flags(device, queue, num_uninserted).await;

        // Pass 4: Scatter non-inserted vertices using compact map
        let mut encoder2 = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder2.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("compact_scatter"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.compact_scatter_pipeline);
            pass.set_bind_group(0, Some(&self.pipelines.compact_scatter_bind_group), &[]);
            pass.dispatch_workgroups(div_ceil(num_uninserted, 256), 1, 1);
        }

        // Return encoder and new_count for caller
        // new_count = number of vertices remaining after compaction
        (encoder2, new_count)
    }

    /// GPU prefix sum on compaction flags.
    /// Returns new_count (number of vertices that will remain after compaction).
    async fn dispatch_gpu_prefix_sum_on_flags(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        num_elements: u32,
    ) -> u32 {
        let vec_size = (num_elements + 3) / 4;
        let thread_blocks = (vec_size + 1023) / 1024;

        // Update params for pack/unpack
        queue.write_buffer(
            &self.pipelines.pack_flags_params,
            0,
            bytemuck::cast_slice(&[num_elements, 0, 0, 0]),
        );
        queue.write_buffer(
            &self.buffers.prefix_sum_info,
            0,
            bytemuck::cast_slice(&[num_elements, vec_size, thread_blocks, 0]),
        );
        queue.write_buffer(
            &self.pipelines.unpack_compact_params,
            0,
            bytemuck::cast_slice(&[num_elements, 0, 0, 0]),
        );

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gpu_prefix_sum_on_flags"),
        });

        // Pass 1: Pack flags (u32) → vec4 for GPU prefix sum
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pack_flags"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.pack_flags_pipeline);
            pass.set_bind_group(0, Some(&self.pipelines.pack_flags_bind_group), &[]);
            pass.dispatch_workgroups((vec_size + 255) / 256, 1, 1);
        }

        // Pass 2: Reduce
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("reduce"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.reduce_pipeline);
            pass.set_bind_group(0, Some(&self.pipelines.reduce_bind_group), &[]);
            pass.dispatch_workgroups(thread_blocks, 1, 1);
        }

        // Pass 3: Spine scan
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("spine_scan"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.spine_scan_pipeline);
            pass.set_bind_group(0, Some(&self.pipelines.spine_scan_bind_group), &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        // Pass 4: Downsweep
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("downsweep"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.downsweep_pipeline);
            pass.set_bind_group(0, Some(&self.pipelines.downsweep_bind_group), &[]);
            pass.dispatch_workgroups(thread_blocks, 1, 1);
        }

        // Pass 5: Unpack vec4 → u32 (result in prefix_sum_data buffer as INCLUSIVE)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("unpack_compact"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.unpack_compact_pipeline);
            pass.set_bind_group(0, Some(&self.pipelines.unpack_compact_bind_group), &[]);
            pass.dispatch_workgroups((vec_size + 255) / 256, 1, 1);
        }

        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });

        // Pass 6: Convert inclusive → exclusive prefix sum
        queue.write_buffer(
            &self.pipelines.inclusive_to_exclusive_params,
            0,
            bytemuck::cast_slice(&[num_elements, 0, 0, 0]),
        );

        let mut encoder2 = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("inclusive_to_exclusive"),
        });
        {
            let mut pass = encoder2.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("inclusive_to_exclusive"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.inclusive_to_exclusive_pipeline);
            pass.set_bind_group(0, Some(&self.pipelines.inclusive_to_exclusive_bind_group), &[]);
            pass.dispatch_workgroups((num_elements + 255) / 256, 1, 1);
        }

        // OPTIMIZATION: Read only last element (8 bytes instead of potentially MB)
        // Copy last elements to staging buffer before submitting
        let last_idx = (num_elements - 1) as u64;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("last_element_staging"),
            size: 8,  // 2 u32s (prefix_sum + flag)
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Copy last prefix_sum element
        encoder2.copy_buffer_to_buffer(
            &self.buffers.prefix_sum_data,
            last_idx * 4,
            &staging,
            0,
            4,
        );

        // Copy last flag element
        encoder2.copy_buffer_to_buffer(
            &self.buffers.compaction_flags,
            last_idx * 4,
            &staging,
            4,
            4,
        );

        queue.submit(Some(encoder2.finish()));
        device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });

        // Read only 8 bytes from staging buffer
        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });

        let data = slice.get_mapped_range();
        let values: &[u32] = bytemuck::cast_slice(&data);
        let new_count = values[0] + values[1];  // prefix_sum[last] + flag[last]

        drop(data);
        staging.unmap();

        new_count
    }

    /// Dispatch relocate points fast.
    /// total_flips: Total number of accumulated flips (for partial binding of flip_arr)
    pub fn dispatch_relocate_points_fast(&self, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue, num_uninserted: u32, total_flips: u32) {
        queue.write_buffer(&self.pipelines.relocate_points_fast_params, 0, bytemuck::cast_slice(&[num_uninserted, 0u32, 0u32, 0u32]));
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("relocate_points_fast"), timestamp_writes: None });
        pass.set_pipeline(&self.pipelines.relocate_points_fast_pipeline);

        // Create dynamic bind group
        let flip_arr_size = if self.use_partial_binding && total_flips > 0 {
            // relocate_points_fast needs to walk FULL accumulated flip chain
            // Bind [0, total_flips) not just current batch
            // CRITICAL: Cap to actual buffer capacity (max_flips = max_tets/2)
            let requested_size = (total_flips as u64) * 32; // FlipItem = 32 bytes
            let max_flip_arr_size = (self.max_flips as u64) * 32;
            let capped_size = requested_size.min(max_flip_arr_size);
            Some(wgpu::BufferSize::new(capped_size).unwrap())
        } else {
            None
        };

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("relocate_points_fast_bg_dynamic"),
            layout: &self.pipelines.relocate_points_fast_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.buffers.uninserted.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.buffers.vert_tet.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.buffers.tet_to_flip.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.buffers.flip_arr,
                        offset: 0,
                        size: flip_arr_size,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.buffers.points.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.pipelines.relocate_points_fast_params.as_entire_binding(),
                },
            ],
        });
        pass.set_bind_group(0, Some(&bind_group), &[]);

        pass.dispatch_workgroups(div_ceil(num_uninserted, 256), 1, 1);
    }

    /// Compact tetras: remove dead tets and rebuild arrays compactly.
    /// This is a critical post-processing step called before extracting final output.
    /// Port of GpuDel::compactTetras() from GpuDelaunay.cu:1362-1402
    ///
    /// After compaction:
    /// - All alive tets are moved to indices 0..new_tet_num-1
    /// - All dead tets are removed
    /// - Adjacency is updated to use new compacted indices
    /// GPU-accelerated inclusive prefix sum for TET_ALIVE flags.
    ///
    /// Performs transform_inclusive_scan equivalent to CUDA:
    ///   1. Transform: tet_info → binary 0/1 (TET_ALIVE bit)
    ///   2. Reduce-Then-Scan: 3-pass GPU prefix sum
    ///   3. Returns new_tet_num from last element
    ///
    /// Leaves result in buffers.prefix_sum_data for downstream shaders.
    pub async fn dispatch_gpu_prefix_sum(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        total_tet_num: u32,
    ) -> u32 {
        log::info!("[GPU_PREFIX_SUM] Starting for {} tets", total_tet_num);

        let vec_size = (total_tet_num + 3) / 4;  // Round up for vec4 packing
        let thread_blocks = (vec_size + 1023) / 1024;  // VEC_PART_SIZE = 1024 vec4

        // Update prefix_sum_info uniform
        queue.write_buffer(
            &self.pipelines.transform_params,
            0,
            bytemuck::cast_slice(&[total_tet_num, 0, 0, 0]),
        );
        queue.write_buffer(
            &self.buffers.prefix_sum_info,
            0,
            bytemuck::cast_slice(&[total_tet_num, vec_size, thread_blocks, 0]),
        );
        queue.write_buffer(
            &self.pipelines.unpack_params,
            0,
            bytemuck::cast_slice(&[total_tet_num, 0, 0, 0]),
        );

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gpu_prefix_sum"),
        });

        // Pass 1: Transform tet_info to binary vec4
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("transform_tet_alive"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.transform_pipeline);
            pass.set_bind_group(0, Some(&self.pipelines.transform_bind_group), &[]);
            pass.dispatch_workgroups((vec_size + 255) / 256, 1, 1);
        }

        // Pass 2: Reduce
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("reduce"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.reduce_pipeline);
            pass.set_bind_group(0, Some(&self.pipelines.reduce_bind_group), &[]);
            pass.dispatch_workgroups(thread_blocks, 1, 1);
        }

        // Pass 3: Spine Scan (single workgroup)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("spine_scan"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.spine_scan_pipeline);
            pass.set_bind_group(0, Some(&self.pipelines.spine_scan_bind_group), &[]);
            pass.dispatch_workgroups(1, 1, 1);  // Single workgroup!
        }

        // Pass 4: Downsweep
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("downsweep"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.downsweep_pipeline);
            pass.set_bind_group(0, Some(&self.pipelines.downsweep_bind_group), &[]);
            pass.dispatch_workgroups(thread_blocks, 1, 1);
        }

        // Pass 5: Unpack vec4 to u32
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("unpack_vec4_to_u32"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.unpack_pipeline);
            pass.set_bind_group(0, Some(&self.pipelines.unpack_bind_group), &[]);
            pass.dispatch_workgroups((vec_size + 255) / 256, 1, 1);
        }

        // Submit all passes
        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });

        // Readback ONLY the last element to get new_tet_num
        let last_idx = total_tet_num - 1;
        let offset = (last_idx as u64) * 4;
        let result: Vec<u32> = self.buffers.read_buffer_range_as(
            device,
            queue,
            &self.buffers.prefix_sum_data,
            offset,
            1,  // Read only 1 u32
        ).await;

        let new_tet_num = result[0];
        log::info!("[GPU_PREFIX_SUM] Complete: {} alive tets ({:.1}%)",
                   new_tet_num, 100.0 * new_tet_num as f32 / total_tet_num as f32);

        new_tet_num
    }

    /// - The tetInfoVec can be set to all TET_ALIVE (all tets in range are alive)
    pub async fn compact_tetras(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        total_tet_num: u32,
    ) -> u32 {
        log::info!("[COMPACT] Starting tet compaction for {} tets", total_tet_num);

        // GPU prefix sum: transform_inclusive_scan on TET_ALIVE flags
        let new_tet_num = self.dispatch_gpu_prefix_sum(device, queue, total_tet_num).await;

        log::info!("[COMPACT] Active tets: {} / {}", new_tet_num, total_tet_num);

        if new_tet_num == total_tet_num {
            log::info!("[COMPACT] No dead tets, skipping compaction");
            return new_tet_num;
        }

        if new_tet_num == 0 {
            log::warn!("[COMPACT] WARNING: No alive tets found!");
            return 0;
        }

        let mut encoder = device.create_command_encoder(&Default::default());

        // Step 1: Collect free slots (dead tet indices)
        self.dispatch_collect_free_slots(&mut encoder, queue, new_tet_num);

        // Step 2: Make compact map (old index → new index mapping)
        self.dispatch_make_compact_map(&mut encoder, queue, new_tet_num, total_tet_num);

        // Step 3: Compact tets (physically move alive tets to front)
        self.dispatch_compact_tets(&mut encoder, queue, new_tet_num, total_tet_num);

        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });

        // Debug: Read back prefix_sum to verify correctness
        if total_tet_num <= 500 {  // Only for small meshes to avoid spam
            let prefix_sample: Vec<u32> = self.buffers.read_buffer_as(
                device, queue, &self.buffers.prefix_sum_data,
                (new_tet_num as usize).min(10) + (total_tet_num as usize - new_tet_num as usize).min(10)
            ).await;

            eprintln!("[COMPACT_DEBUG] Prefix (first 10): {:?}",
                     &prefix_sample[..prefix_sample.len().min(10)]);
        }

        log::info!("[COMPACT] ✓ Compaction complete, {} alive tets", new_tet_num);

        new_tet_num
    }
}

fn div_ceil(a: u32, b: u32) -> u32 {
    (a + b - 1) / b
}
