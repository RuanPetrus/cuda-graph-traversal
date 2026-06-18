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

__global__ static void bfs_root_init_kernel(u64 root, i64 *pred, i32 *frontier,
		i32 *frontier_count, i32 *next_frontier_count, i32 npes) {
	if (blockIdx.x != 0 || threadIdx.x != 0) return;

	*frontier_count = 0;
	*next_frontier_count = 0;
	i32 my_pe = nvshmem_my_pe();
	if (vertex_owner(root, npes) == my_pe) {
		u64 root_local = vertex_local(root, npes);
		pred[root_local] = (i64)root;
		frontier[0] = (i32)root_local;
		*frontier_count = 1;
	}
}

__global__ static void bfs_expand_kernel(const u64 *rowstarts, const u64 *column, i64 *pred,
		const i32 *frontier, i32 *next_frontier, const i32 *frontier_count,
		i32 *next_frontier_count, i32 npes) {
	i32 tid = blockIdx.x * blockDim.x + threadIdx.x;
	i32 stride = blockDim.x * gridDim.x;
	i32 count = *frontier_count;
	i32 my_pe = nvshmem_my_pe();

	for (i32 idx = tid; idx < count; idx += stride) {
		u64 src_local = (u64)frontier[idx];
		u64 src_global = src_local * (u64)npes + (u64)my_pe;
		for (u64 edge_idx = rowstarts[src_local]; edge_idx < rowstarts[src_local + 1]; ++edge_idx) {
			u64 dst_global = column[edge_idx];
			i32 dst_pe = vertex_owner(dst_global, npes);
			u64 dst_local = vertex_local(dst_global, npes);
			i64 old = nvshmem_int64_atomic_compare_swap(&pred[dst_local], -1, (i64)src_global, dst_pe);
			if (old == -1) {
				i32 pos = nvshmem_int_atomic_fetch_add(next_frontier_count, 1, dst_pe);
				nvshmem_int_p(&next_frontier[pos], (i32)dst_local, dst_pe);
			}
		}
	}
	nvshmem_quiet();
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
		i32 threads_per_block = 256;
		u64 blocks_needed = INT_CEIL(tg->edges_size, (u64)threads_per_block);
		i32 blocks = blocks_needed > 65535 ? 65535 : (i32)blocks_needed;
		void *args[] = {&tg->edges, &tg->edges_size, &degrees, &ws->rank_size};
		i32 status = nvshmemx_collective_launch((const void *)compute_degrees_kernel, dim3(blocks), dim3(threads_per_block), args, 0, 0);
		if (status != 0) {
			ERROR("NVSHMEM collective launch failed for degree computation");
			worker_abort(ws, 1);
		}
		CHECK_CUDA(cudaDeviceSynchronize());
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
		i32 threads_per_block = 256;
		u64 blocks_needed = INT_CEIL(tg->edges_size, (u64)threads_per_block);
		i32 blocks = blocks_needed > 65535 ? 65535 : (i32)blocks_needed;
		void *args[] = {&tg->edges, &tg->weights, &tg->edges_size, &degrees, &g.rowstarts, &g.column, &g.weights, &ws->rank_size};
		i32 status = nvshmemx_collective_launch((const void *)fill_csr_kernel, dim3(blocks), dim3(threads_per_block), args, 0, 0);
		if (status != 0) {
			ERROR("NVSHMEM collective launch failed for CSR fill");
			worker_abort(ws, 1);
		}
		CHECK_CUDA(cudaDeviceSynchronize());
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

	i32 current = 0;
	i32 next = 1;
	i32 rank_size = ws->rank_size;

	i64 *pred = bfs->pred;
	i32 *frontier_current = bfs->frontier[current];
	i32 *frontier_next = bfs->frontier[next];
	i32 *frontier_count_current = bfs->frontier_count[current];
	i32 *frontier_count_next = bfs->frontier_count[next];
	void *init_args[] = {&root, &pred, &frontier_current, &frontier_count_current, &frontier_count_next, &rank_size};
	i32 status = nvshmemx_collective_launch((const void *)bfs_root_init_kernel, dim3(1), dim3(1), init_args, 0, 0);
	if (status != 0) {
		ERROR("NVSHMEM collective launch failed for BFS root init");
		worker_abort(ws, 1);
	}
	CHECK_CUDA(cudaDeviceSynchronize());
	nvshmem_barrier_all();

	i32 local_count = 0;
	i64 local_count_64 = 0;
	i64 global_count = 0;
	CHECK_CUDA(cudaMemcpy(&local_count, bfs->frontier_count[current], sizeof(local_count), cudaMemcpyDeviceToHost));
	local_count_64 = local_count;
	MPI_Allreduce(&local_count_64, &global_count, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);

	dim3 expand_block(256);
	dim3 expand_grid(128);
	while (global_count > 0) {
		CHECK_CUDA(cudaMemset(bfs->frontier_count[next], 0, sizeof(i32)));
		CHECK_CUDA(cudaDeviceSynchronize());
		nvshmem_barrier_all();

		u64 *rowstarts = g->rowstarts;
		u64 *column = g->column;
		pred = bfs->pred;
		frontier_current = bfs->frontier[current];
		frontier_next = bfs->frontier[next];
		frontier_count_current = bfs->frontier_count[current];
		frontier_count_next = bfs->frontier_count[next];
		rank_size = ws->rank_size;
		void *expand_args[] = {&rowstarts, &column, &pred, &frontier_current, &frontier_next,
				&frontier_count_current, &frontier_count_next, &rank_size};
		status = nvshmemx_collective_launch((const void *)bfs_expand_kernel, expand_grid, expand_block, expand_args, 0, 0);
		if (status != 0) {
			ERROR("NVSHMEM collective launch failed for BFS expansion");
			worker_abort(ws, 1);
		}
		CHECK_CUDA(cudaDeviceSynchronize());
		nvshmem_barrier_all();

		CHECK_CUDA(cudaMemcpy(&local_count, bfs->frontier_count[next], sizeof(local_count), cudaMemcpyDeviceToHost));
		local_count_64 = local_count;
		MPI_Allreduce(&local_count_64, &global_count, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);

		i32 tmp = current;
		current = next;
		next = tmp;
	}
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
		i32 status = nvshmemx_collective_launch((const void *)bfs_dist_propagate_kernel, dim3(blocks), dim3(threads_per_block), args, 0, 0);
		if (status != 0) {
			ERROR("NVSHMEM collective launch failed for BFS distance propagation");
			worker_abort(ws, 1);
		}
		CHECK_CUDA(cudaDeviceSynchronize());
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

void oned_graph_free(Oned_Graph *g) {
	if (!g) return;
	*g = (Oned_Graph){0};
}
