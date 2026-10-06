#include "traversal.h"
#include <limits.h>
#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>

__host__ __device__ static i32 vertex_owner(u64 v, i32 npes) {
	return (i32)(v % (u64)npes);
}

__host__ __device__ static u64 vertex_local(u64 v, i32 npes) {
	return v / (u64)npes;
}

__global__ static void compute_degrees_kernel(Packed_Edge *edges, u64 edge_count, u64 *degrees, i32 npes) {
	u64 tid = (u64)blockIdx.x * blockDim.x + threadIdx.x;
	u64 stride = (u64)blockDim.x * gridDim.x;

	for (u64 edge_idx = tid; edge_idx < edge_count; edge_idx += stride) {
		u64 v0 = edges[edge_idx].v0;
		u64 v1 = edges[edge_idx].v1;
		if (v0 == v1) continue;

		i32 pe0 = vertex_owner(v0, npes);
		u64 local0 = vertex_local(v0, npes);
		nvshmem_uint64_atomic_add(&degrees[local0], 1, pe0);

		i32 pe1 = vertex_owner(v1, npes);
		u64 local1 = vertex_local(v1, npes);
		nvshmem_uint64_atomic_add(&degrees[local1], 1, pe1);
	}

	nvshmem_quiet();
}

// TODO(ruan): Use a good psum kernel
__global__ static void compute_rowstarts_kernel(u64 *degrees, u64 nlocalverts, u64 *rowstarts) {
	if (blockIdx.x != 0 || threadIdx.x != 0) return;

	u64 sum = 0;
	rowstarts[0] = 0;
	for (u64 vertex_idx = 0; vertex_idx < nlocalverts; ++vertex_idx) {
		sum += degrees[vertex_idx];
		rowstarts[vertex_idx + 1] = sum;
	}
}

__device__ static void append_directed_edge(u64 src, u64 dst, f32 weight, u64 *degrees, u64 *rowstarts, u64 *column, f32 *weights, i32 npes) {
	i32 pe = vertex_owner(src, npes);
	u64 local = vertex_local(src, npes);
	u64 offset = nvshmem_uint64_atomic_fetch_add(&degrees[local], (u64)1, pe);
	u64 rowstart = nvshmem_uint64_g(&rowstarts[local], pe);
	u64 pos = rowstart + offset;
	nvshmem_uint64_p(&column[pos], dst, pe);
	nvshmem_float_p(&weights[pos], weight, pe);
}

__global__ static void fill_csr_kernel(Packed_Edge *edges, f32 *edge_weights, u64 edge_count, u64 *degrees, u64 *rowstarts, u64 *column, f32 *weights, i32 npes) {
	u64 tid = (u64)blockIdx.x * blockDim.x + threadIdx.x;
	u64 stride = (u64)blockDim.x * gridDim.x;

	for (u64 edge_idx = tid; edge_idx < edge_count; edge_idx += stride) {
		u64 v0 = edges[edge_idx].v0;
		u64 v1 = edges[edge_idx].v1;
		if (v0 == v1) continue;

		f32 weight = edge_weights[edge_idx];
		append_directed_edge(v0, v1, weight, degrees, rowstarts, column, weights, npes);
		append_directed_edge(v1, v0, weight, degrees, rowstarts, column, weights, npes);
	}

	nvshmem_quiet();
}

__global__ static void bfs_dist_init_kernel(i64 *pred, i64 *dist, u64 nlocalverts, u64 root, i32 rank, i32 npes) {
	u64 tid = (u64)blockIdx.x * blockDim.x + threadIdx.x;
	u64 stride = (u64)blockDim.x * gridDim.x;

	for (u64 local_vertex = tid; local_vertex < nlocalverts; local_vertex += stride) {
		u64 global_vertex = local_vertex * (u64)npes + (u64)rank;
		dist[local_vertex] = pred[local_vertex] == -1 ? -1 : INT64_MAX;
		if (global_vertex == root) {
			dist[local_vertex] = 0;
		}
	}
}

__global__ static void bfs_dist_propagate_kernel(const u64 *rowstarts, const u64 *column, const i64 *pred, i64 *dist,
		u64 nlocalverts, i64 current_level, u64 *new_visits, i32 rank, i32 npes) {
	u64 tid = (u64)blockIdx.x * blockDim.x + threadIdx.x;
	u64 stride = (u64)blockDim.x * gridDim.x;

	for (u64 src_local = tid; src_local < nlocalverts; src_local += stride) {
		if (dist[src_local] != current_level) continue;

		u64 src_global = src_local * (u64)npes + (u64)rank;
		for (u64 edge_idx = rowstarts[src_local]; edge_idx < rowstarts[src_local + 1]; ++edge_idx) {
			u64 dst_global = column[edge_idx];
			i32 dst_pe = vertex_owner(dst_global, npes);
			u64 dst_local = vertex_local(dst_global, npes);
			i64 dst_pred = nvshmem_int64_g(&pred[dst_local], dst_pe);
			if (dst_pred != (i64)src_global) continue;

			i64 old = nvshmem_int64_atomic_compare_swap(&dist[dst_local], INT64_MAX, current_level + 1, dst_pe);
			if (old == INT64_MAX) {
				nvshmem_uint64_atomic_add(new_visits, 1, dst_pe);
			}
		}
	}

	nvshmem_quiet();
}

__device__ static void bfs_grid_sync(u32 *barrier_count, u32 *barrier_sense, i32 block_count) {
	__syncthreads();
	if (threadIdx.x == 0) {
		volatile u32 *sense = barrier_sense;
		u32 old_sense = *sense;
		__threadfence();
		u32 arrived = atomicAdd((unsigned int *)barrier_count, 1);
		if (arrived == (u32)block_count - 1) {
			*barrier_count = 0;
			__threadfence();
			*sense = old_sense + 1;
		} else {
			while (*sense == old_sense) { }
		}
	}
	__syncthreads();
}

__device__ static void bfs_device_sync(u32 *barrier_count, u32 *barrier_sense, i32 block_count) {
	nvshmem_quiet();
	bfs_grid_sync(barrier_count, barrier_sense, block_count);
	if (blockIdx.x == 0 && threadIdx.x == 0) nvshmem_barrier_all();
	bfs_grid_sync(barrier_count, barrier_sense, block_count);
}

__global__ static void bfs_run_kernel(u64 root, const u64 *rowstarts, const u64 *column, i64 *pred,
		i32 *frontier0, i32 *frontier1, i32 *frontier_count0, i32 *frontier_count1,
		u64 *global_frontier_count, u32 *barrier_count, u32 *barrier_sense, i32 block_count, i32 npes) {
	i32 my_pe = nvshmem_my_pe();
	i32 *frontier_current = frontier0;
	i32 *frontier_next = frontier1;
	i32 *frontier_count_current = frontier_count0;
	i32 *frontier_count_next = frontier_count1;

	if (threadIdx.x == 0) {
		if (blockIdx.x == 0) {
		*frontier_count_current = 0;
		*frontier_count_next = 0;
		if (vertex_owner(root, npes) == my_pe) {
			u64 root_local = vertex_local(root, npes);
			pred[root_local] = (i64)root;
			frontier_current[0] = (i32)root_local;
			*frontier_count_current = 1;
		}
		}
	}
	bfs_device_sync(barrier_count, barrier_sense, block_count);

	while (true) {
		if (threadIdx.x == 0 && my_pe == 0) *global_frontier_count = 0;
		bfs_device_sync(barrier_count, barrier_sense, block_count);

		i32 local_count = *frontier_count_current;
		if (blockIdx.x == 0 && threadIdx.x == 0) {
			nvshmem_uint64_atomic_add(global_frontier_count, (u64)local_count, 0);
		}
		bfs_device_sync(barrier_count, barrier_sense, block_count);

		u64 global_count = nvshmem_uint64_g(global_frontier_count, 0);
		__syncthreads();
		if (global_count == 0) break;

		if (blockIdx.x == 0 && threadIdx.x == 0) *frontier_count_next = 0;
		bfs_device_sync(barrier_count, barrier_sense, block_count);

		i32 global_thread = (i32)(blockIdx.x * blockDim.x + threadIdx.x);
		i32 lane = global_thread & 31;
		i32 warp_idx = global_thread >> 5;
		i32 warp_count = ((i32)(blockDim.x * gridDim.x)) >> 5;
		for (i32 idx = warp_idx; idx < local_count; idx += warp_count) {
			u64 src_local = (u64)frontier_current[idx];
			u64 src_global = src_local * (u64)npes + (u64)my_pe;
			for (u64 edge_idx = rowstarts[src_local] + (u64)lane; edge_idx < rowstarts[src_local + 1]; edge_idx += 32) {
				u64 dst_global = column[edge_idx];
				i32 dst_pe = vertex_owner(dst_global, npes);
				u64 dst_local = vertex_local(dst_global, npes);
				i64 old = nvshmem_int64_atomic_compare_swap(&pred[dst_local], -1, (i64)src_global, dst_pe);
				if (old == -1) {
					i32 pos = nvshmem_int_atomic_fetch_add(frontier_count_next, 1, dst_pe);
					nvshmem_int_p(&frontier_next[pos], (i32)dst_local, dst_pe);
				}
			}
		}
		bfs_device_sync(barrier_count, barrier_sense, block_count);

		i32 *frontier_tmp = frontier_current;
		frontier_current = frontier_next;
		frontier_next = frontier_tmp;
		i32 *frontier_count_tmp = frontier_count_current;
		frontier_count_current = frontier_count_next;
		frontier_count_next = frontier_count_tmp;
		bfs_grid_sync(barrier_count, barrier_sense, block_count);
	}
}

static const u32 SSSP_INF_BITS = 0x7f800000U;

__device__ static void sssp_relax(u32 *dist_bits, i64 *pred, i32 *next_frontier, i32 *next_frontier_count,
		i32 *visited, i32 *pred_lock, u64 dst_global, f32 candidate_dist, u64 src_local,
		f32 bucket_max, b32 light_phase, i32 npes) {
	i32 my_pe = nvshmem_my_pe();
	i32 dst_pe = vertex_owner(dst_global, npes);
	u64 dst_local = vertex_local(dst_global, npes);
	i64 src_global = (i64)(src_local * (u64)npes + (u64)my_pe);
	u32 candidate_bits = __float_as_uint(candidate_dist);
	u32 old_bits = nvshmem_uint_g(&dist_bits[dst_local], dst_pe);

	while (candidate_bits < old_bits) {
		u32 previous = nvshmem_uint_atomic_compare_swap(&dist_bits[dst_local], old_bits, candidate_bits, dst_pe);
		if (previous == old_bits) {
			while (nvshmem_int_atomic_compare_swap(&pred_lock[dst_local], 0, 1, dst_pe) != 0) { }
			if (nvshmem_uint_g(&dist_bits[dst_local], dst_pe) == candidate_bits) {
				nvshmem_int64_p(&pred[dst_local], src_global, dst_pe);
			}
			nvshmem_int_p(&pred_lock[dst_local], 0, dst_pe);
			if (light_phase && candidate_dist < bucket_max) {
				i32 was_seen = nvshmem_int_atomic_compare_swap(&visited[dst_local], 0, 1, dst_pe);
				if (was_seen == 0) {
					i32 position = nvshmem_int_atomic_fetch_add(next_frontier_count, 1, dst_pe);
					nvshmem_int_p(&next_frontier[position], (i32)dst_local, dst_pe);
				}
			}
			break;
		}
		old_bits = previous;
	}
}

__global__ static void sssp_clear_kernel(u32 *dist_bits, i64 *pred, f32 *dist, i32 *visited, i32 *pred_lock,
		u64 nlocalverts) {
	u64 tid = (u64)blockIdx.x * blockDim.x + threadIdx.x;
	u64 stride = (u64)blockDim.x * gridDim.x;
	for (u64 i = tid; i < nlocalverts; i += stride) {
		dist_bits[i] = SSSP_INF_BITS;
		pred[i] = -1;
		dist[i] = -1.0f;
		visited[i] = 0;
		pred_lock[i] = 0;
	}
}

__global__ static void sssp_root_init_kernel(u64 root, u32 *dist_bits, i64 *pred, i32 *frontier,
		i32 *frontier_count, i32 *next_frontier_count, i32 npes) {
	if (blockIdx.x != 0 || threadIdx.x != 0) return;
	*frontier_count = 0;
	*next_frontier_count = 0;
	i32 my_pe = nvshmem_my_pe();
	if (vertex_owner(root, npes) == my_pe) {
		u64 root_local = vertex_local(root, npes);
		dist_bits[root_local] = __float_as_uint(0.0f);
		pred[root_local] = (i64)root;
		frontier[0] = (i32)root_local;
		*frontier_count = 1;
	}
}

__global__ static void sssp_clear_light_kernel(i32 *visited, i32 *next_frontier_count, u64 nlocalverts) {
	u64 tid = (u64)blockIdx.x * blockDim.x + threadIdx.x;
	u64 stride = (u64)blockDim.x * gridDim.x;
	if (tid == 0) *next_frontier_count = 0;
	for (u64 i = tid; i < nlocalverts; i += stride) visited[i] = 0;
}

__global__ static void sssp_light_relax_kernel(const u64 *rowstarts, const u64 *column, const f32 *weights,
		u32 *dist_bits, i64 *pred, const i32 *frontier, i32 *next_frontier, const i32 *frontier_count,
		i32 *next_frontier_count, i32 *visited, i32 *pred_lock, f32 delta, f32 bucket_max, i32 npes) {
	u64 tid = (u64)blockIdx.x * blockDim.x + threadIdx.x;
	u64 stride = (u64)blockDim.x * gridDim.x;
	i32 count = *frontier_count;
	for (i32 index = (i32)tid; index < count; index += (i32)stride) {
		u64 src_local = (u64)frontier[index];
		f32 src_dist = __uint_as_float(dist_bits[src_local]);
		for (u64 edge = rowstarts[src_local]; edge < rowstarts[src_local + 1]; ++edge) {
			f32 weight = weights[edge];
			if (weight < delta) {
				sssp_relax(dist_bits, pred, next_frontier, next_frontier_count, visited, pred_lock,
						column[edge], src_dist + weight, src_local, bucket_max, true, npes);
			}
		}
	}
	nvshmem_quiet();
}

__global__ static void sssp_heavy_relax_kernel(const u64 *rowstarts, const u64 *column, const f32 *weights,
		u32 *dist_bits, i64 *pred, i32 *pred_lock, f32 delta, f32 bucket_min, f32 bucket_max,
		u64 nlocalverts, i32 npes) {
	u64 tid = (u64)blockIdx.x * blockDim.x + threadIdx.x;
	u64 stride = (u64)blockDim.x * gridDim.x;
	for (u64 src_local = tid; src_local < nlocalverts; src_local += stride) {
		f32 src_dist = __uint_as_float(dist_bits[src_local]);
		if (src_dist < bucket_min || src_dist >= bucket_max) continue;
		for (u64 edge = rowstarts[src_local]; edge < rowstarts[src_local + 1]; ++edge) {
			f32 weight = weights[edge];
			if (weight >= delta) {
				sssp_relax(dist_bits, pred, 0, 0, 0, pred_lock, column[edge], src_dist + weight,
						src_local, bucket_max, false, npes);
			}
		}
	}
	nvshmem_quiet();
}

__global__ static void sssp_bucket_scan_kernel(const u32 *dist_bits, i32 *frontier, i32 *frontier_count,
		u64 *pending_count, f32 bucket_min, f32 bucket_max, u64 nlocalverts) {
	u64 tid = (u64)blockIdx.x * blockDim.x + threadIdx.x;
	u64 stride = (u64)blockDim.x * gridDim.x;
	for (u64 i = tid; i < nlocalverts; i += stride) {
		if (dist_bits[i] == SSSP_INF_BITS) continue;
		f32 distance = __uint_as_float(dist_bits[i]);
		if (distance >= bucket_min) {
			atomicAdd((unsigned long long *)pending_count, 1ULL);
			if (distance < bucket_max) {
				i32 position = atomicAdd(frontier_count, 1);
				frontier[position] = (i32)i;
			}
		}
	}
}

__global__ static void sssp_export_dist_kernel(const u32 *dist_bits, f32 *dist, u64 nlocalverts) {
	u64 tid = (u64)blockIdx.x * blockDim.x + threadIdx.x;
	u64 stride = (u64)blockDim.x * gridDim.x;
	for (u64 i = tid; i < nlocalverts; i += stride) {
		u32 bits = dist_bits[i];
		dist[i] = bits == SSSP_INF_BITS ? -1.0f : __uint_as_float(bits);
	}
}

Oned_Graph oned_graph_from_tuple_graph(Worker_State *ws, Tuple_Graph* tg, u64 nglobalverts) {
	Oned_Graph g = {0};
	g.nglobalverts = nglobalverts;
	g.nlocalverts = INT_CEIL(MAX((i64)nglobalverts - ws->rank, 0), ws->rank_size);

	// Computing degrees
	u64 max_degree_count = 0;
	u64 *degrees = worker_arena_push_array(ws, g.nlocalverts, u64, &max_degree_count);
	CHECK_CUDA(cudaMemset(degrees, 0, max_degree_count * sizeof(u64)));
	if (tg->edges_size > 0) {
		void *args[] = {&tg->edges, &tg->edges_size, &degrees, &ws->rank_size};
		worker_kernel_launch(ws, (const void *)compute_degrees_kernel, args, tg->edges_size, 256, "degree computation");
	}
	nvshmem_barrier_all();

	// Computing row start
	// TODO(ruan): Use a good psum kernel
	u64 max_rowstart_count = 0;
	g.rowstarts = worker_arena_push_array(ws, g.nlocalverts + 1, u64, &max_rowstart_count);
	CHECK_CUDA(cudaMemset(g.rowstarts, 0, max_rowstart_count * sizeof(u64)));
	compute_rowstarts_kernel<<<1, 1>>>(degrees, g.nlocalverts, g.rowstarts);
	CHECK_CUDA(cudaGetLastError());
	CHECK_CUDA(cudaDeviceSynchronize());
	nvshmem_barrier_all();

	// Computing column and weights
	CHECK_CUDA(cudaMemcpy(&g.nlocaledges, g.rowstarts + g.nlocalverts, sizeof(g.nlocaledges), cudaMemcpyDeviceToHost));
	CHECK_CUDA(cudaMemset(degrees, 0, max_degree_count * sizeof(u64)));

	u64 max_column_count = 0;
	g.column = worker_arena_push_array(ws, g.nlocaledges, u64, &max_column_count);
	CHECK_CUDA(cudaMemset(g.column, 0, max_column_count * sizeof(u64)));

	u64 max_weight_count = 0;
	g.weights = worker_arena_push_array(ws, g.nlocaledges, f32, &max_weight_count);
	CHECK_CUDA(cudaMemset(g.weights, 0, max_weight_count * sizeof(f32)));
	if (max_weight_count != max_column_count) {
		ERROR("Mismatched CSR column and weight allocation sizes");
		worker_abort(ws, 1);
	}
	// Use the degree buffer, and the rowstart buffer to compute column and weights
	if (tg->edges_size > 0) {
		void *args[] = {&tg->edges, &tg->weights, &tg->edges_size, &degrees, &g.rowstarts, &g.column, &g.weights, &ws->rank_size};
		worker_kernel_launch(ws, (const void *)fill_csr_kernel, args, tg->edges_size, 256, "CSR fill");
	}
	nvshmem_barrier_all();
	return g;
}

b32 oned_graph_is_vertex_isolated(Worker_State *ws, const Oned_Graph *g, u64 global_vertex) {
	if (global_vertex >= g->nglobalverts) {
		ERROR("Vertex %llu is outside graph vertex range [0, %llu)",
				(unsigned long long)global_vertex,
				(unsigned long long)g->nglobalverts);
		worker_abort(ws, 1);
	}

	i32 owner = vertex_owner(global_vertex, ws->rank_size);
	u64 local = vertex_local(global_vertex, ws->rank_size);
	i32 isolated = 0;
	if (ws->rank == owner) {
		u64 rowstart[2];
		CHECK_CUDA(cudaMemcpy(rowstart, g->rowstarts + local, sizeof(rowstart), cudaMemcpyDeviceToHost));
		isolated = rowstart[0] == rowstart[1];
	}

	MPI_Allreduce(MPI_IN_PLACE, &isolated, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
	return isolated != 0;
}

Bfs_State oned_graph_bfs_create(Worker_State *ws, const Oned_Graph *g) {
	Bfs_State bfs = {0};

	bfs.pred = worker_arena_push_array(ws, g->nlocalverts, i64, &bfs.max_nlocalverts);

	u64 max_frontier_count = 0;
	bfs.frontier[0] = worker_arena_push_array(ws, g->nlocalverts, i32, &max_frontier_count);
	if (max_frontier_count != bfs.max_nlocalverts) {
		ERROR("Mismatched BFS frontier allocation size");
		worker_abort(ws, 1);
	}

	bfs.frontier[1] = worker_arena_push_array(ws, g->nlocalverts, i32, &max_frontier_count);
	if (max_frontier_count != bfs.max_nlocalverts) {
		ERROR("Mismatched BFS frontier allocation size");
		worker_abort(ws, 1);
	}

	u64 max_count_slots = 0;
	bfs.frontier_count[0] = worker_arena_push_array(ws, 1, i32, &max_count_slots);
	if (max_count_slots != 1) {
		ERROR("Mismatched BFS frontier count allocation size");
		worker_abort(ws, 1);
	}
	bfs.frontier_count[1] = worker_arena_push_array(ws, 1, i32, &max_count_slots);
	if (max_count_slots != 1) {
		ERROR("Mismatched BFS frontier count allocation size");
		worker_abort(ws, 1);
	}
	u64 max_global_count_slots = 0;
	bfs.global_frontier_count = worker_arena_push_array(ws, 1, u64, &max_global_count_slots);
	if (max_global_count_slots != 1) {
		ERROR("Mismatched BFS global frontier count allocation size");
		worker_abort(ws, 1);
	}
	u64 max_barrier_count_slots = 0;
	bfs.barrier_count = worker_arena_push_array(ws, 1, u32, &max_barrier_count_slots);
	if (max_barrier_count_slots != 1) {
		ERROR("Mismatched BFS barrier count allocation size");
		worker_abort(ws, 1);
	}
	bfs.barrier_sense = worker_arena_push_array(ws, 1, u32, &max_barrier_count_slots);
	if (max_barrier_count_slots != 1) {
		ERROR("Mismatched BFS barrier sense allocation size");
		worker_abort(ws, 1);
	}

	oned_graph_bfs_clear(ws, &bfs);

	return bfs;
}

void oned_graph_bfs_clear(Worker_State *ws, Bfs_State *bfs) {
	(void)ws;
	CHECK_CUDA(cudaMemset(bfs->pred, 0xff, bfs->max_nlocalverts * sizeof(i64)));
	if (bfs->dist) CHECK_CUDA(cudaMemset(bfs->dist, 0xff, bfs->max_nlocalverts * sizeof(i64)));
	CHECK_CUDA(cudaMemset(bfs->frontier[0], 0, bfs->max_nlocalverts * sizeof(i32)));
	CHECK_CUDA(cudaMemset(bfs->frontier[1], 0, bfs->max_nlocalverts * sizeof(i32)));
	CHECK_CUDA(cudaMemset(bfs->frontier_count[0], 0, sizeof(i32)));
	CHECK_CUDA(cudaMemset(bfs->frontier_count[1], 0, sizeof(i32)));
	CHECK_CUDA(cudaMemset(bfs->global_frontier_count, 0, sizeof(u64)));
	CHECK_CUDA(cudaMemset(bfs->barrier_count, 0, sizeof(u32)));
	CHECK_CUDA(cudaMemset(bfs->barrier_sense, 0, sizeof(u32)));
	CHECK_CUDA(cudaDeviceSynchronize());
	nvshmem_barrier_all();
}

void oned_graph_bfs_run(Worker_State *ws, const Oned_Graph *g, Bfs_State *bfs, u64 root) {
	if (root >= g->nglobalverts) {
		ERROR("BFS root %llu is outside graph vertex range [0, %llu)",
				(unsigned long long)root,
				(unsigned long long)g->nglobalverts);
		worker_abort(ws, 1);
	}

	i32 rank_size = ws->rank_size;
	u64 *rowstarts = g->rowstarts;
	u64 *column = g->column;
	i64 *pred = bfs->pred;
	i32 *frontier0 = bfs->frontier[0];
	i32 *frontier1 = bfs->frontier[1];
	i32 *frontier_count0 = bfs->frontier_count[0];
	i32 *frontier_count1 = bfs->frontier_count[1];
	u64 *global_frontier_count = bfs->global_frontier_count;
	u32 *barrier_count = bfs->barrier_count;
	u32 *barrier_sense = bfs->barrier_sense;
	i32 block_count = 1;
	void *args[] = {&root, &rowstarts, &column, &pred, &frontier0, &frontier1, &frontier_count0, &frontier_count1, &global_frontier_count, &barrier_count, &barrier_sense, &block_count, &rank_size};
	block_count = worker_kernel_block_count(ws, (const void *)bfs_run_kernel, args, bfs->max_nlocalverts, 256, "BFS run");
	worker_kernel_launch(ws, (const void *)bfs_run_kernel, args, bfs->max_nlocalverts, 256, "BFS run");
	nvshmem_barrier_all();
}

i64 *bfs_compute_dist_from_pred(Worker_State *ws, const Oned_Graph *g, Bfs_State *bfs, u64 root) {
	if (root >= g->nglobalverts) {
		ERROR("BFS root %llu is outside graph vertex range [0, %llu)",
				(unsigned long long)root,
				(unsigned long long)g->nglobalverts);
		worker_abort(ws, 1);
	}

	if (!bfs->dist) {
		u64 max_dist_count = 0;
		bfs->dist = worker_arena_push_array(ws, g->nlocalverts, i64, &max_dist_count);
		if (max_dist_count != bfs->max_nlocalverts) {
			ERROR("Mismatched BFS distance allocation size");
			worker_abort(ws, 1);
		}
	}

	u64 max_new_visit_count = 0;
	u64 *new_visits = worker_arena_push_array(ws, 1, u64, &max_new_visit_count);
	if (max_new_visit_count != 1) {
		ERROR("Mismatched BFS distance new visit allocation size");
		worker_abort(ws, 1);
	}

	i32 threads_per_block = 256;
	u64 blocks_needed = INT_CEIL(g->nlocalverts, (u64)threads_per_block);
	i32 blocks = blocks_needed > 65535 ? 65535 : (i32)blocks_needed;
	if (blocks == 0) blocks = 1;

	bfs_dist_init_kernel<<<blocks, threads_per_block>>>(bfs->pred, bfs->dist, g->nlocalverts, root, ws->rank, ws->rank_size);
	CHECK_CUDA(cudaGetLastError());
	CHECK_CUDA(cudaDeviceSynchronize());
	nvshmem_barrier_all();

	for (i64 current_level = 0;; ++current_level) {
		CHECK_CUDA(cudaMemset(new_visits, 0, sizeof(u64)));
		CHECK_CUDA(cudaDeviceSynchronize());
		nvshmem_barrier_all();

		u64 *rowstarts = g->rowstarts;
		u64 *column = g->column;
		i64 *pred = bfs->pred;
		i64 *dist = bfs->dist;
		u64 nlocalverts = g->nlocalverts;
		i32 rank = ws->rank;
		i32 rank_size = ws->rank_size;
		void *args[] = {&rowstarts, &column, &pred, &dist, &nlocalverts, &current_level, &new_visits, &rank, &rank_size};
		worker_kernel_launch(ws, (const void *)bfs_dist_propagate_kernel, args, g->nlocalverts, 256, "BFS distance propagation");
		nvshmem_barrier_all();

		u64 local_new_visits = 0;
		u64 global_new_visits = 0;
		CHECK_CUDA(cudaMemcpy(&local_new_visits, new_visits, sizeof(local_new_visits), cudaMemcpyDeviceToHost));
		MPI_Allreduce(&local_new_visits, &global_new_visits, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
		if (global_new_visits == 0) break;
	}

	return bfs->dist;
}

void oned_graph_bfs_destroy(Bfs_State *bfs) {
	if (!bfs) return;
	*bfs = (Bfs_State){0};
}

Sssp_State oned_graph_sssp_create(Worker_State *ws, const Oned_Graph *g) {
	Sssp_State sssp = {0};

	sssp.pred = worker_arena_push_array(ws, g->nlocalverts, i64, &sssp.max_nlocalverts);
	u64 max_count = 0;
	sssp.dist = worker_arena_push_array(ws, g->nlocalverts, f32, &max_count);
	if (max_count != sssp.max_nlocalverts) {
		ERROR("Mismatched SSSP distance allocation size");
		worker_abort(ws, 1);
	}
	sssp.dist_bits = worker_arena_push_array(ws, g->nlocalverts, u32, &max_count);
	if (max_count != sssp.max_nlocalverts) {
		ERROR("Mismatched SSSP distance-bit allocation size");
		worker_abort(ws, 1);
	}
	sssp.frontier[0] = worker_arena_push_array(ws, g->nlocalverts, i32, &max_count);
	if (max_count != sssp.max_nlocalverts) {
		ERROR("Mismatched SSSP frontier allocation size");
		worker_abort(ws, 1);
	}
	sssp.frontier[1] = worker_arena_push_array(ws, g->nlocalverts, i32, &max_count);
	if (max_count != sssp.max_nlocalverts) {
		ERROR("Mismatched SSSP frontier allocation size");
		worker_abort(ws, 1);
	}
	sssp.visited = worker_arena_push_array(ws, g->nlocalverts, i32, &max_count);
	if (max_count != sssp.max_nlocalverts) {
		ERROR("Mismatched SSSP visited allocation size");
		worker_abort(ws, 1);
	}
	sssp.pred_lock = worker_arena_push_array(ws, g->nlocalverts, i32, &max_count);
	if (max_count != sssp.max_nlocalverts) {
		ERROR("Mismatched SSSP predecessor-lock allocation size");
		worker_abort(ws, 1);
	}
	sssp.frontier_count[0] = worker_arena_push_array(ws, 1, i32, &max_count);
	if (max_count != 1) {
		ERROR("Mismatched SSSP frontier-count allocation size");
		worker_abort(ws, 1);
	}
	sssp.frontier_count[1] = worker_arena_push_array(ws, 1, i32, &max_count);
	if (max_count != 1) {
		ERROR("Mismatched SSSP frontier-count allocation size");
		worker_abort(ws, 1);
	}
	sssp.pending_count = worker_arena_push_array(ws, 1, u64, &max_count);
	if (max_count != 1) {
		ERROR("Mismatched SSSP pending-count allocation size");
		worker_abort(ws, 1);
	}

	oned_graph_sssp_clear(ws, &sssp);
	return sssp;
}

void oned_graph_sssp_clear(Worker_State *ws, Sssp_State *sssp) {
	u64 nlocalverts = sssp->max_nlocalverts;
	void *args[] = {&sssp->dist_bits, &sssp->pred, &sssp->dist, &sssp->visited, &sssp->pred_lock, &nlocalverts};
	worker_kernel_launch(ws, (const void *)sssp_clear_kernel, args, nlocalverts, 256, "SSSP clear");
	CHECK_CUDA(cudaMemset(sssp->frontier[0], 0, nlocalverts * sizeof(i32)));
	CHECK_CUDA(cudaMemset(sssp->frontier[1], 0, nlocalverts * sizeof(i32)));
	CHECK_CUDA(cudaMemset(sssp->frontier_count[0], 0, sizeof(i32)));
	CHECK_CUDA(cudaMemset(sssp->frontier_count[1], 0, sizeof(i32)));
	CHECK_CUDA(cudaMemset(sssp->pending_count, 0, sizeof(u64)));
	CHECK_CUDA(cudaDeviceSynchronize());
	nvshmem_barrier_all();
}

void oned_graph_sssp_run(Worker_State *ws, const Oned_Graph *g, Sssp_State *sssp, u64 root) {
	if (root >= g->nglobalverts) {
		ERROR("SSSP root %llu is outside graph vertex range [0, %llu)",
				(unsigned long long)root, (unsigned long long)g->nglobalverts);
		worker_abort(ws, 1);
	}

	const f32 delta = 0.1f;
	f32 bucket_min = 0.0f;
	f32 bucket_max = delta;
	i32 current = 0;
	i32 next = 1;
	i32 npes = ws->rank_size;
	u64 nlocalverts = g->nlocalverts;
	u64 *rowstarts = g->rowstarts;
	u64 *column = g->column;
	f32 *weights = g->weights;

	void *init_args[] = {&root, &sssp->dist_bits, &sssp->pred, &sssp->frontier[current],
			&sssp->frontier_count[current], &sssp->frontier_count[next], &npes};
	worker_kernel_launch(ws, (const void *)sssp_root_init_kernel, init_args, 1, 1, "SSSP root initialization");
	nvshmem_barrier_all();

	i64 global_count = 1;
	while (global_count != 0) {
		i32 local_count = 0;
		CHECK_CUDA(cudaMemcpy(&local_count, sssp->frontier_count[current], sizeof(local_count), cudaMemcpyDeviceToHost));
		i64 local_count_64 = local_count;
		MPI_Allreduce(&local_count_64, &global_count, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);

		while (global_count != 0) {
			void *clear_args[] = {&sssp->visited, &sssp->frontier_count[next], &nlocalverts};
			worker_kernel_launch(ws, (const void *)sssp_clear_light_kernel, clear_args, nlocalverts, 256, "SSSP light-frontier clear");
			nvshmem_barrier_all();

			void *light_args[] = {&rowstarts, &column, &weights, &sssp->dist_bits, &sssp->pred,
					&sssp->frontier[current], &sssp->frontier[next], &sssp->frontier_count[current],
					&sssp->frontier_count[next], &sssp->visited, &sssp->pred_lock, (void *)&delta,
					&bucket_max, &npes};
			worker_kernel_launch(ws, (const void *)sssp_light_relax_kernel, light_args, nlocalverts, 256, "SSSP light relaxation");
			nvshmem_barrier_all();

			i32 temporary = current;
			current = next;
			next = temporary;
			CHECK_CUDA(cudaMemcpy(&local_count, sssp->frontier_count[current], sizeof(local_count), cudaMemcpyDeviceToHost));
			local_count_64 = local_count;
			MPI_Allreduce(&local_count_64, &global_count, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
		}

		void *heavy_args[] = {&rowstarts, &column, &weights, &sssp->dist_bits, &sssp->pred,
				&sssp->pred_lock, (void *)&delta, &bucket_min, &bucket_max, &nlocalverts, &npes};
		worker_kernel_launch(ws, (const void *)sssp_heavy_relax_kernel, heavy_args, nlocalverts, 256, "SSSP heavy relaxation");
		nvshmem_barrier_all();

		bucket_min = bucket_max;
		bucket_max += delta;
		CHECK_CUDA(cudaMemset(sssp->frontier_count[current], 0, sizeof(i32)));
		CHECK_CUDA(cudaMemset(sssp->pending_count, 0, sizeof(u64)));
		CHECK_CUDA(cudaDeviceSynchronize());
		void *scan_args[] = {&sssp->dist_bits, &sssp->frontier[current], &sssp->frontier_count[current],
				&sssp->pending_count, &bucket_min, &bucket_max, &nlocalverts};
		worker_kernel_launch(ws, (const void *)sssp_bucket_scan_kernel, scan_args, nlocalverts, 256, "SSSP bucket scan");
		nvshmem_barrier_all();

		u64 local_pending = 0;
		CHECK_CUDA(cudaMemcpy(&local_pending, sssp->pending_count, sizeof(local_pending), cudaMemcpyDeviceToHost));
		MPI_Allreduce(&local_pending, &global_count, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
	}

	void *export_args[] = {&sssp->dist_bits, &sssp->dist, &nlocalverts};
	worker_kernel_launch(ws, (const void *)sssp_export_dist_kernel, export_args, nlocalverts, 256, "SSSP distance export");
	nvshmem_barrier_all();
}

void oned_graph_sssp_destroy(Sssp_State *sssp) {
	if (!sssp) return;
	*sssp = (Sssp_State){0};
}

void oned_graph_free(Oned_Graph *g) {
	if (!g) return;
	*g = (Oned_Graph){0};
}
