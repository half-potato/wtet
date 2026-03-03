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

    /// Dispatch pick winner.
    pub fn dispatch_pick_winner(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
    ) {
        queue.write_buffer(
            &self.pipelines.pick_params,
            0,
            bytemuck::cast_slice(&[self.max_tets, 0u32, 0u32, 0u32]),
        );

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("pick_winner"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipelines.pick_pipeline);
        pass.set_bind_group(0, Some(&self.pipelines.pick_bind_group), &[]);
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
    pub fn dispatch_reset_votes(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("reset_votes"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipelines.reset_votes_pipeline);
        pass.set_bind_group(0, Some(&self.pipelines.reset_votes_bind_group), &[]);
        pass.dispatch_workgroups(div_ceil(self.max_tets, 64), 1, 1);
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

    // ==================== STUB DISPATCH FUNCTIONS (COMMENTED OUT) ====================
    // TODO: Implement these kernels and uncomment when ready
    /*
    pub fn dispatch_collect_free_slots(&self, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue, num_tets: u32) {
        // ...stub...
    }

    pub fn dispatch_make_compact_map(&self, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue, new_tet_num: u32, total_tet_num: u32) {
        // ...stub...
    }

    pub fn dispatch_compact_tets(&self, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue, new_tet_num: u32, total_tet_num: u32) {
        // ...stub...
    }

    pub fn dispatch_mark_special_tets(&self, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue) {
        // ...stub...
    }

    pub fn dispatch_update_flip_trace(&self, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue, org_flip_num: u32, flip_num: u32) {
        // ...stub...
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
    */

    /// Dispatch update opp (CRITICAL - flip adjacency updates).
    pub fn dispatch_update_opp(&self, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue, org_flip_num: u32, flip_num: u32) {
        queue.write_buffer(&self.pipelines.update_opp_params, 0, bytemuck::cast_slice(&[org_flip_num, flip_num, 0u32, 0u32]));
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("update_opp"), timestamp_writes: None });
        pass.set_pipeline(&self.pipelines.update_opp_pipeline);
        pass.set_bind_group(0, Some(&self.pipelines.update_opp_bind_group), &[]);
        pass.dispatch_workgroups(div_ceil(flip_num, 64), 1, 1);
    }

    /// Dispatch mark rejected flips (flip validation).
    pub fn dispatch_mark_rejected_flips(&self, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue, act_tet_num: u32, vote_offset: i32, compact_mode: bool) {
        queue.write_buffer(&self.pipelines.mark_rejected_flips_params, 0, bytemuck::cast_slice(&[act_tet_num, vote_offset as u32, if compact_mode { 1u32 } else { 0u32 }, 0u32]));
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("mark_rejected_flips"), timestamp_writes: None });
        pass.set_pipeline(&self.pipelines.mark_rejected_flips_pipeline);
        pass.set_bind_group(0, Some(&self.pipelines.mark_rejected_flips_bind_group), &[]);
        pass.dispatch_workgroups(div_ceil(act_tet_num, 64), 1, 1);
    }

    /// Dispatch check delaunay fast (flip voting).
    pub fn dispatch_check_delaunay_fast(&self, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue, act_tet_num: u32, vote_offset: u32) {
        queue.write_buffer(&self.pipelines.check_delaunay_fast_params, 0, bytemuck::cast_slice(&[act_tet_num, vote_offset, 0u32, 0u32]));
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("check_delaunay_fast"), timestamp_writes: None });
        pass.set_pipeline(&self.pipelines.check_delaunay_fast_pipeline);
        pass.set_bind_group(0, Some(&self.pipelines.check_delaunay_fast_bind_group), &[]);
        pass.dispatch_workgroups(div_ceil(act_tet_num, 64), 1, 1);
    }

    /// Dispatch allocate flip23 slot.
    pub fn dispatch_allocate_flip23_slot(&self, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue, flip_num: u32, inf_idx: u32, tet_num: u32) {
        queue.write_buffer(&self.pipelines.allocate_flip23_slot_params, 0, bytemuck::cast_slice(&[flip_num, inf_idx, tet_num, 0u32]));
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("allocate_flip23_slot"), timestamp_writes: None });
        pass.set_pipeline(&self.pipelines.allocate_flip23_slot_pipeline);
        pass.set_bind_group(0, Some(&self.pipelines.allocate_flip23_slot_bind_group), &[]);
        pass.dispatch_workgroups(div_ceil(flip_num, 64), 1, 1);
    }

    /// Dispatch compact if negative (two-pass compaction).
    /// Returns the compacted count via reading counter[0] after both passes.
    pub fn dispatch_compact_if_negative(&self, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue, input_size: u32) {
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

    /// Dispatch relocate points fast.
    pub fn dispatch_relocate_points_fast(&self, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue, num_uninserted: u32) {
        queue.write_buffer(&self.pipelines.relocate_points_fast_params, 0, bytemuck::cast_slice(&[num_uninserted, 0u32, 0u32, 0u32]));
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("relocate_points_fast"), timestamp_writes: None });
        pass.set_pipeline(&self.pipelines.relocate_points_fast_pipeline);
        pass.set_bind_group(0, Some(&self.pipelines.relocate_points_fast_bind_group), &[]);
        pass.dispatch_workgroups(div_ceil(num_uninserted, 64), 1, 1);
    }
}

fn div_ceil(a: u32, b: u32) -> u32 {
    (a + b - 1) / b
}
