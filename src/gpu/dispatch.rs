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

    // ==================== NEW DISPATCH FUNCTIONS (14 KERNELS) ====================

    /// Dispatch collect free slots (compaction).
    pub fn dispatch_collect_free_slots(&self, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue, num_tets: u32) {
        queue.write_buffer(&self.pipelines.collect_free_slots_params, 0, bytemuck::cast_slice(&[num_tets, 0u32, 0u32, 0u32]));
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("collect_free_slots"), timestamp_writes: None });
        pass.set_pipeline(&self.pipelines.collect_free_slots_pipeline);
        pass.set_bind_group(0, Some(&self.pipelines.collect_free_slots_bind_group), &[]);
        pass.dispatch_workgroups(div_ceil(num_tets, 64), 1, 1);
    }

    /// Dispatch make compact map (compaction).
    pub fn dispatch_make_compact_map(&self, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue, new_tet_num: u32, total_tet_num: u32) {
        queue.write_buffer(&self.pipelines.make_compact_map_params, 0, bytemuck::cast_slice(&[new_tet_num, total_tet_num, 0u32, 0u32]));
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("make_compact_map"), timestamp_writes: None });
        pass.set_pipeline(&self.pipelines.make_compact_map_pipeline);
        pass.set_bind_group(0, Some(&self.pipelines.make_compact_map_bind_group), &[]);
        pass.dispatch_workgroups(div_ceil(total_tet_num - new_tet_num, 64), 1, 1);
    }

    /// Dispatch compact tets (compaction).
    pub fn dispatch_compact_tets(&self, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue, new_tet_num: u32, total_tet_num: u32) {
        queue.write_buffer(&self.pipelines.compact_tets_params, 0, bytemuck::cast_slice(&[new_tet_num, total_tet_num, 0u32, 0u32]));
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("compact_tets"), timestamp_writes: None });
        pass.set_pipeline(&self.pipelines.compact_tets_pipeline);
        pass.set_bind_group(0, Some(&self.pipelines.compact_tets_bind_group), &[]);
        pass.dispatch_workgroups(div_ceil(total_tet_num - new_tet_num, 64), 1, 1);
    }

    /// Dispatch mark special tets (flip management).
    pub fn dispatch_mark_special_tets(&self, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue) {
        queue.write_buffer(&self.pipelines.mark_special_tets_params, 0, bytemuck::cast_slice(&[self.max_tets, 0u32, 0u32, 0u32]));
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("mark_special_tets"), timestamp_writes: None });
        pass.set_pipeline(&self.pipelines.mark_special_tets_pipeline);
        pass.set_bind_group(0, Some(&self.pipelines.mark_special_tets_bind_group), &[]);
        pass.dispatch_workgroups(div_ceil(self.max_tets, 64), 1, 1);
    }

    /// Dispatch update flip trace (flip management).
    pub fn dispatch_update_flip_trace(&self, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue, org_flip_num: u32, flip_num: u32) {
        queue.write_buffer(&self.pipelines.update_flip_trace_params, 0, bytemuck::cast_slice(&[org_flip_num, flip_num, 0u32, 0u32]));
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("update_flip_trace"), timestamp_writes: None });
        pass.set_pipeline(&self.pipelines.update_flip_trace_pipeline);
        pass.set_bind_group(0, Some(&self.pipelines.update_flip_trace_bind_group), &[]);
        pass.dispatch_workgroups(div_ceil(flip_num, 64), 1, 1);
    }

    /// Dispatch update block vert free list (block allocation).
    pub fn dispatch_update_block_vert_free_list(&self, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue, ins_num: u32, old_ins_num: u32) {
        queue.write_buffer(&self.pipelines.update_block_vert_free_list_params, 0, bytemuck::cast_slice(&[ins_num, old_ins_num, 0u32, 0u32]));
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("update_block_vert_free_list"), timestamp_writes: None });
        pass.set_pipeline(&self.pipelines.update_block_vert_free_list_pipeline);
        pass.set_bind_group(0, Some(&self.pipelines.update_block_vert_free_list_bind_group), &[]);
        pass.dispatch_workgroups(div_ceil(ins_num * 8, 64), 1, 1); // MEAN_VERTEX_DEGREE = 8
    }

    /// Dispatch update block opp tet idx (block allocation).
    pub fn dispatch_update_block_opp_tet_idx(&self, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue, old_inf_block_idx: u32, new_inf_block_idx: u32, old_tet_num: u32) {
        queue.write_buffer(&self.pipelines.update_block_opp_tet_idx_params, 0, bytemuck::cast_slice(&[old_inf_block_idx, new_inf_block_idx, old_tet_num, 0u32]));
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("update_block_opp_tet_idx"), timestamp_writes: None });
        pass.set_pipeline(&self.pipelines.update_block_opp_tet_idx_pipeline);
        pass.set_bind_group(0, Some(&self.pipelines.update_block_opp_tet_idx_bind_group), &[]);
        pass.dispatch_workgroups(div_ceil(old_tet_num, 64), 1, 1);
    }

    /// Dispatch shift inf free idx (index shifting).
    pub fn dispatch_shift_inf_free_idx(&self, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue, inf_idx: u32, start: u32, shift: u32) {
        queue.write_buffer(&self.pipelines.shift_inf_free_idx_params, 0, bytemuck::cast_slice(&[inf_idx, start, shift, 0u32]));
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("shift_inf_free_idx"), timestamp_writes: None });
        pass.set_pipeline(&self.pipelines.shift_inf_free_idx_pipeline);
        pass.set_bind_group(0, Some(&self.pipelines.shift_inf_free_idx_bind_group), &[]);
        // Dispatched based on vert_free_arr[inf_idx] - would need to read that first
        pass.dispatch_workgroups(1, 1, 1); // Placeholder
    }

    /// Dispatch update tet idx (index shifting).
    pub fn dispatch_update_tet_idx(&self, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue, old_inf_block_idx: u32, new_inf_block_idx: u32, vec_size: u32) {
        queue.write_buffer(&self.pipelines.update_tet_idx_params, 0, bytemuck::cast_slice(&[old_inf_block_idx, new_inf_block_idx, vec_size, 0u32]));
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("update_tet_idx"), timestamp_writes: None });
        pass.set_pipeline(&self.pipelines.update_tet_idx_pipeline);
        pass.set_bind_group(0, Some(&self.pipelines.update_tet_idx_bind_group), &[]);
        pass.dispatch_workgroups(div_ceil(vec_size, 64), 1, 1);
    }

    /// Dispatch shift opp tet idx (index shifting).
    pub fn dispatch_shift_opp_tet_idx(&self, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue, tet_num: u32, start: u32, shift: u32) {
        queue.write_buffer(&self.pipelines.shift_opp_tet_idx_params, 0, bytemuck::cast_slice(&[tet_num, start, shift, 0u32]));
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("shift_opp_tet_idx"), timestamp_writes: None });
        pass.set_pipeline(&self.pipelines.shift_opp_tet_idx_pipeline);
        pass.set_bind_group(0, Some(&self.pipelines.shift_opp_tet_idx_bind_group), &[]);
        pass.dispatch_workgroups(div_ceil(tet_num, 64), 1, 1);
    }

    /// Dispatch shift tet idx (index shifting).
    pub fn dispatch_shift_tet_idx(&self, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue, vec_size: u32, start: u32, shift: u32) {
        queue.write_buffer(&self.pipelines.shift_tet_idx_params, 0, bytemuck::cast_slice(&[vec_size, start, shift, 0u32]));
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("shift_tet_idx"), timestamp_writes: None });
        pass.set_pipeline(&self.pipelines.shift_tet_idx_pipeline);
        pass.set_bind_group(0, Some(&self.pipelines.shift_tet_idx_bind_group), &[]);
        pass.dispatch_workgroups(div_ceil(vec_size, 64), 1, 1);
    }

    /// Dispatch make reverse map (utility).
    pub fn dispatch_make_reverse_map(&self, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue, ins_vert_num: u32, num: u32) {
        queue.write_buffer(&self.pipelines.make_reverse_map_params, 0, bytemuck::cast_slice(&[ins_vert_num, num, 0u32, 0u32]));
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("make_reverse_map"), timestamp_writes: None });
        pass.set_pipeline(&self.pipelines.make_reverse_map_pipeline);
        pass.set_bind_group(0, Some(&self.pipelines.make_reverse_map_bind_group), &[]);
        pass.dispatch_workgroups(div_ceil(ins_vert_num, 64), 1, 1);
    }

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
}

fn div_ceil(a: u32, b: u32) -> u32 {
    (a + b - 1) / b
}
