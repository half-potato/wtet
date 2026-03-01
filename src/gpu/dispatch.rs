use super::GpuState;

/// Dispatch helpers that encode and submit compute passes.
impl GpuState {
    /// Dispatch the init kernel (single workgroup).
    pub fn dispatch_init(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("init"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipelines.init_pipeline);
        pass.set_bind_group(0, Some(&self.pipelines.init_bind_group), &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    /// Dispatch point location for `num_uninserted` points.
    pub fn dispatch_locate(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        num_uninserted: u32,
    ) {
        // Update params
        queue.write_buffer(
            &self.pipelines.locate_params,
            0,
            bytemuck::cast_slice(&[num_uninserted, 512u32, 0u32, 0u32]),
        );

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("locate"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipelines.locate_pipeline);
        pass.set_bind_group(0, Some(&self.pipelines.locate_bind_group), &[]);
        pass.dispatch_workgroups(div_ceil(num_uninserted, 64), 1, 1);
    }

    /// Dispatch vote for point.
    pub fn dispatch_vote(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        num_uninserted: u32,
    ) {
        queue.write_buffer(
            &self.pipelines.vote_params,
            0,
            bytemuck::cast_slice(&[num_uninserted, 0u32, 0u32, 0u32]),
        );

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("vote"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipelines.vote_pipeline);
        pass.set_bind_group(0, Some(&self.pipelines.vote_bind_group), &[]);
        pass.dispatch_workgroups(div_ceil(num_uninserted, 64), 1, 1);
    }

    /// Dispatch pick winner (iterates over uninserted vertices).
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
        pass.dispatch_workgroups(div_ceil(num_uninserted, 64), 1, 1);
    }

    /// Dispatch build insert list (scans tets to build insert_list from tet_vert winners).
    pub fn dispatch_build_insert_list(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
    ) {
        queue.write_buffer(
            &self.pipelines.build_insert_list_params,
            0,
            bytemuck::cast_slice(&[self.max_tets, 0u32, 0u32, 0u32]),
        );

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("build_insert_list"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipelines.build_insert_list_pipeline);
        pass.set_bind_group(0, Some(&self.pipelines.build_insert_list_bind_group), &[]);
        pass.dispatch_workgroups(div_ceil(self.max_tets, 64), 1, 1);
    }

    /// Dispatch mark split (marks tets being split for concurrent detection).
    pub fn dispatch_mark_split(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        num_insertions: u32,
    ) {
        queue.write_buffer(
            &self.pipelines.mark_split_params,
            0,
            bytemuck::cast_slice(&[num_insertions, 0u32, 0u32, 0u32]),
        );

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("mark_split"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipelines.mark_split_pipeline);
        pass.set_bind_group(0, Some(&self.pipelines.mark_split_bind_group), &[]);
        pass.dispatch_workgroups(div_ceil(num_insertions, 64), 1, 1);
    }

    /// Dispatch split points (update vert_tet for vertices whose tets are splitting).
    pub fn dispatch_split_points(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        num_uninserted: u32,
    ) {
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
        pass.set_bind_group(0, Some(&self.pipelines.split_points_bind_group), &[]);
        pass.dispatch_workgroups(div_ceil(num_uninserted, 64), 1, 1);
    }

    /// Dispatch split tetra.
    pub fn dispatch_split(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        num_insertions: u32,
    ) {
        queue.write_buffer(
            &self.pipelines.split_params,
            0,
            bytemuck::cast_slice(&[num_insertions, 0u32, 0u32, 0u32]),
        );

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("split"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipelines.split_pipeline);
        pass.set_bind_group(0, Some(&self.pipelines.split_bind_group), &[]);
        pass.dispatch_workgroups(div_ceil(num_insertions, 64), 1, 1);
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
        pass.dispatch_workgroups(div_ceil(num_uninserted, 64), 1, 1);
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
        use_alternate: bool,
    ) {
        queue.write_buffer(
            &self.pipelines.flip_params,
            0,
            bytemuck::cast_slice(&[queue_size, 0u32, 0u32, 0u32]),
        );

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("flip"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipelines.flip_pipeline);
        let bg = if use_alternate {
            &self.pipelines.flip_bind_group_b
        } else {
            &self.pipelines.flip_bind_group
        };
        pass.set_bind_group(0, Some(bg), &[]);
        pass.dispatch_workgroups(div_ceil(queue_size, 64), 1, 1);
    }

    /// Dispatch reset votes.
    pub fn dispatch_reset_votes(&self, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue, num_uninserted: u32) {
        queue.write_buffer(
            &self.pipelines.reset_votes_params,
            0,
            bytemuck::cast_slice(&[num_uninserted, self.max_tets, 0u32, 0u32]),
        );

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("reset_votes"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipelines.reset_votes_pipeline);
        pass.set_bind_group(0, Some(&self.pipelines.reset_votes_bind_group), &[]);
        // Dispatch enough workgroups for both arrays
        let max_size = std::cmp::max(num_uninserted, self.max_tets);
        pass.dispatch_workgroups(div_ceil(max_size, 64), 1, 1);
    }

    /// Dispatch gather failed vertices.
    pub fn dispatch_gather(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gather"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipelines.gather_pipeline);
        pass.set_bind_group(0, Some(&self.pipelines.gather_bind_group), &[]);
        pass.dispatch_workgroups(div_ceil(self.max_tets, 64), 1, 1);
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
}

fn div_ceil(a: u32, b: u32) -> u32 {
    (a + b - 1) / b
}
