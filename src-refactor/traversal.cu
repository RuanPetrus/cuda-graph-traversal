#include "traversal.h"
#include <nvshmem.h>
#include <nvshmemx.h>

__device__ static i32 vertex_owner_device(u64 v, i32 npes) {
	return (i32)(v % (u64)npes);
}

__device__ static u64 vertex_local_device(u64 v, i32 npes) {
	return v / (u64)npes;
}

__global__ static void compute_degrees_kernel(Packed_Edge *edges, u64 edge_count, u64 *degrees, i32 npes) {
	u64 tid = (u64)blockIdx.x * blockDim.x + threadIdx.x;
	u64 stride = (u64)blockDim.x * gridDim.x;

	for (u64 edge_idx = tid; edge_idx < edge_count; edge_idx += stride) {
		u64 v0 = edges[edge_idx].v0;
		u64 v1 = edges[edge_idx].v1;
		if (v0 == v1) continue;

		i32 pe0 = vertex_owner_device(v0, npes);
		u64 local0 = vertex_local_device(v0, npes);
		nvshmem_uint64_atomic_add(&degrees[local0], 1, pe0);

		i32 pe1 = vertex_owner_device(v1, npes);
		u64 local1 = vertex_local_device(v1, npes);
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
	i32 pe = vertex_owner_device(src, npes);
	u64 local = vertex_local_device(src, npes);
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

void oned_graph_free(Oned_Graph *g) {
	if (!g) return;
	*g = (Oned_Graph){0};
}
