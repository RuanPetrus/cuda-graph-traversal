#include "worker.h"
#include "graph_generation.h"
#include "traversal.h"
#include "visualization.h"

#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#define GPU_MEMORY_SIZE GIGABYTE(2)
#define BFS_ROOT_COUNT 64

static i32 generate_bfs_roots(Worker_State *ws, const Oned_Graph *g, u64 seed1, u64 seed2, u64 *roots, i32 max_roots) {
	u64 counter = 0;
	i32 root_count = 0;
	for (; root_count < max_roots; ++root_count) {
		u64 root = 0;
		b32 found_root = false;
		while (true) {
			f64 d[2];
			make_random_numbers(2, seed1, seed2, (i64)counter, d);
			root = (u64)((d[0] + d[1]) * (f64)g->nglobalverts) % g->nglobalverts;
			counter += 2;
			if (counter > 2 * g->nglobalverts) break;

			b32 duplicate = false;
			for (i32 i = 0; i < root_count; ++i) {
				if (root == roots[i]) {
					duplicate = true;
					break;
				}
			}
			if (duplicate) continue;
			if (oned_graph_is_vertex_isolated(ws, g, root)) continue;

			found_root = true;
			break;
		}

		if (!found_root) break;
		roots[root_count] = root;
	}
	return root_count;
}

__host__ __device__ static i32 validation_vertex_owner(u64 global_vertex, i32 npes) {
	return (i32)(global_vertex % (u64)npes);
}

__host__ __device__ static u64 validation_vertex_local(u64 global_vertex, i32 npes) {
	return global_vertex / (u64)npes;
}

__global__ static void bfs_validate_init_kernel(const i64 *pred, const i64 *dist, i32 *confirmed, u64 *error_count,
		u64 nlocalverts, u64 nglobalverts, u64 root, i32 rank, i32 npes) {
	u64 tid = (u64)blockIdx.x * blockDim.x + threadIdx.x;
	u64 stride = (u64)blockDim.x * gridDim.x;

	for (u64 local_vertex = tid; local_vertex < nlocalverts; local_vertex += stride) {
		u64 global_vertex = local_vertex * (u64)npes + (u64)rank;
		confirmed[local_vertex] = 0;

		// Step 1: predecessor values must be either -1 or a valid global vertex.
		i64 p = pred[local_vertex];
		if (p != -1 && (p < 0 || (u64)p >= nglobalverts)) {
			nvshmem_uint64_atomic_add(error_count, 1, rank);
		}

		// Step 2: the root must point to itself and have distance zero.
		if (global_vertex == root) {
			if (p != (i64)root || dist[local_vertex] != 0) {
				nvshmem_uint64_atomic_add(error_count, 1, rank);
			} else {
				confirmed[local_vertex] = 1;
			}
		}
	}
	nvshmem_quiet();
}

__global__ static void bfs_validate_edges_kernel(const u64 *rowstarts, const u64 *column, const i64 *pred, const i64 *dist,
		i32 *confirmed, u64 *error_count, u64 nlocalverts, i32 rank, i32 npes) {
	u64 tid = (u64)blockIdx.x * blockDim.x + threadIdx.x;
	u64 stride = (u64)blockDim.x * gridDim.x;

	for (u64 src_local = tid; src_local < nlocalverts; src_local += stride) {
		u64 src_global = src_local * (u64)npes + (u64)rank;
		i64 src_pred = pred[src_local];
		i64 src_dist = dist[src_local];

		for (u64 edge_idx = rowstarts[src_local]; edge_idx < rowstarts[src_local + 1]; ++edge_idx) {
			u64 dst_global = column[edge_idx];
			// Step 3: self edges are ignored by graph construction and should not be present here.
			if (dst_global == src_global) {
				nvshmem_uint64_atomic_add(error_count, 1, rank);
				continue;
			}

			i32 dst_pe = validation_vertex_owner(dst_global, npes);
			u64 dst_local = validation_vertex_local(dst_global, npes);
			i64 dst_pred = nvshmem_int64_g(&pred[dst_local], dst_pe);
			i64 dst_dist = nvshmem_int64_g(&dist[dst_local], dst_pe);

			// Step 4: no edge may connect a visited vertex to an unvisited vertex.
			if ((src_dist == -1 && dst_dist != -1) || (src_dist != -1 && dst_dist == -1)) {
				nvshmem_uint64_atomic_add(error_count, 1, rank);
			}
			// Step 5: BFS distances on an edge may differ by at most one level.
			if (src_dist >= 0 && dst_dist >= 0 && (src_dist + 1 < dst_dist || dst_dist + 1 < src_dist)) {
				nvshmem_uint64_atomic_add(error_count, 1, rank);
			}

			// Step 6: mark vertices whose claimed predecessor is confirmed by this edge.
			if (src_pred == (i64)dst_global) {
				confirmed[src_local] = 1;
			}
			if (dst_pred == (i64)src_global) {
				nvshmem_int_p(&confirmed[dst_local], 1, dst_pe);
			}
		}
	}
	nvshmem_quiet();
}

__global__ static void bfs_validate_confirmed_kernel(const i64 *pred, const i64 *dist, const i32 *confirmed, u64 *error_count,
		u64 nlocalverts, u64 root, i32 rank, i32 npes) {
	u64 tid = (u64)blockIdx.x * blockDim.x + threadIdx.x;
	u64 stride = (u64)blockDim.x * gridDim.x;

	for (u64 local_vertex = tid; local_vertex < nlocalverts; local_vertex += stride) {
		u64 global_vertex = local_vertex * (u64)npes + (u64)rank;
		if (pred[local_vertex] == -1) continue;
		// Step 7: every reached non-root vertex must have a resolved distance and a confirmed predecessor edge.
		if (dist[local_vertex] == INT64_MAX || (global_vertex != root && !confirmed[local_vertex])) {
			nvshmem_uint64_atomic_add(error_count, 1, rank);
		}
	}
	nvshmem_quiet();
}

static b32 oned_graph_bfs_validate(Worker_State *ws, Arena *temp_arena, const Oned_Graph *g, Bfs_State *bfs, u64 root) {
	// Validation setup: derive BFS distances from the distributed predecessor tree.
	i64 *dist = bfs_compute_dist_from_pred(ws, g, bfs, root);

	// Validation setup: allocate symmetric buffers for predecessor confirmation and error counting.
	i32 *confirmed = arena_push_array(temp_arena, bfs->max_nlocalverts, i32);
	u64 *error_count = arena_push_array(temp_arena, 1, u64);

	CHECK_CUDA(cudaMemset(error_count, 0, sizeof(u64)));
	CHECK_CUDA(cudaDeviceSynchronize());
	nvshmem_barrier_all();

	// Validation setup: use one local thread domain per PE over that PE's owned vertices.
	i32 threads_per_block = 256;
	u64 blocks_needed = INT_CEIL(g->nlocalverts, (u64)threads_per_block);
	i32 blocks = blocks_needed > 65535 ? 65535 : (i32)blocks_needed;
	if (blocks == 0) blocks = 1;

	// Validation pass 1: clear confirmations, check predecessor range, and validate the root.
	bfs_validate_init_kernel<<<blocks, threads_per_block>>>(bfs->pred, dist, confirmed, error_count,
			g->nlocalverts, g->nglobalverts, root, ws->rank, ws->rank_size);
	CHECK_CUDA(cudaGetLastError());
	CHECK_CUDA(cudaDeviceSynchronize());
	nvshmem_barrier_all();

	u64 *rowstarts = g->rowstarts;
	u64 *column = g->column;
	i64 *pred = bfs->pred;
	u64 nlocalverts = g->nlocalverts;
	i32 rank = ws->rank;
	i32 rank_size = ws->rank_size;
	void *edge_args[] = {&rowstarts, &column, &pred, &dist, &confirmed, &error_count, &nlocalverts, &rank, &rank_size};
	// Validation pass 2: scan distributed CSR edges, compare remote pred/dist, and confirm predecessor edges.
	i32 status = nvshmemx_collective_launch((const void *)bfs_validate_edges_kernel, dim3(blocks), dim3(threads_per_block), edge_args, 0, 0);
	if (status != 0) {
		ERROR("NVSHMEM collective launch failed for BFS validation edges");
		worker_abort(ws, 1);
	}
	CHECK_CUDA(cudaDeviceSynchronize());
	nvshmem_barrier_all();

	// Validation pass 3: reject reached vertices with unresolved distances or unconfirmed predecessors.
	bfs_validate_confirmed_kernel<<<blocks, threads_per_block>>>(pred, dist, confirmed, error_count,
			nlocalverts, root, rank, rank_size);
	CHECK_CUDA(cudaGetLastError());
	CHECK_CUDA(cudaDeviceSynchronize());
	nvshmem_barrier_all();

	// Validation reduction: combine per-PE error counters into a global pass/fail result.
	u64 local_errors = 0;
	u64 global_errors = 0;
	CHECK_CUDA(cudaMemcpy(&local_errors, error_count, sizeof(local_errors), cudaMemcpyDeviceToHost));
	MPI_Allreduce(&local_errors, &global_errors, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
	if (global_errors && ws->rank == 0) {
		fprintf(stderr, "Validation Error: BFS validation found %llu error(s)\n", (unsigned long long)global_errors);
	}
	return global_errors == 0;
}

i32 main(i32 argc, char **argv) {
	Worker_State ws = worker_init(argc, argv, GPU_MEMORY_SIZE);

	i32 SCALE = 16;
	i32 edgefactor = 16;
	if (argc >= 2) SCALE = atoi(argv[1]);
	if (argc >= 3) edgefactor = atoi(argv[2]);
	if (argc <= 1 || argc >= 4 || SCALE == 0 || edgefactor == 0) {
		if (ws.rank == 0) {
			fprintf(stderr, "Usage: %s SCALE edgefactor\n  SCALE = log_2(# vertices) [integer, required]\n  edgefactor = (# edges) / (# vertices) = .5 * (average vertex degree) [integer, defaults to 16]\n(Random number seed and Kronecker initiator are in graph500_runner.c)\n", argv[0]);
		}
		worker_abort(&ws, 1);
	}
	u64 seed1 = 2, seed2 = 3;

	Tuple_Graph tg = {0};
	tg.nglobaledges = (i64)(edgefactor) << SCALE;

	u32 seed[5];
	make_mrg_seed(seed1, seed2, seed);
	{
		i64 edge_per_rank_count = INT_CEIL(tg.nglobaledges, ws.rank_size);
		i64 start_edge_index = MIN(edge_per_rank_count * ws.rank, tg.nglobaledges);
		i64 end_edge_index = MIN(start_edge_index + edge_per_rank_count, tg.nglobaledges);
		generate_kronecker_range(&ws, seed, SCALE, start_edge_index, end_edge_index, &tg);
		tuple_graph_dump(&tg, start_edge_index); // Use to check against old implementation
	}

	Oned_Graph g = oned_graph_from_tuple_graph(&ws, &tg, (u64)1 << SCALE);
	u64 bfs_roots[BFS_ROOT_COUNT];
	i32 bfs_root_count = generate_bfs_roots(&ws, &g, seed1, seed2, bfs_roots, BFS_ROOT_COUNT);
	if (ws.rank == 0) {
		fprintf(stderr, "bfs_roots:                      %d\n", bfs_root_count);
	}

	if (!getenv("SKIP_BFS") && bfs_root_count > 0) {
		Bfs_State bfs = oned_graph_bfs_create(&ws, &g);
		Arena validation_arena = arena_create_gpu(bfs.max_nlocalverts * sizeof(i32) + sizeof(u64) + KILOBYTE(4));

		oned_graph_bfs_clear(&ws, &bfs);
		oned_graph_bfs_run(&ws, &g, &bfs, bfs_roots[0]); // Warm-up, like the original Graph500 runner.

		for (i32 bfs_root_idx = 0; bfs_root_idx < bfs_root_count; ++bfs_root_idx) {
			u64 root = bfs_roots[bfs_root_idx];
			if (ws.rank == 0) fprintf(stderr, "Running BFS %d\n", bfs_root_idx);

			oned_graph_bfs_clear(&ws, &bfs);
			oned_graph_bfs_run(&ws, &g, &bfs, root);
			if (getenv("WRITE_DOT")) {
				bfs_compute_dist_from_pred(&ws, &g, &bfs, root);
				bfs_write_dot_files_for_root(&ws, &g, &bfs, root, bfs_root_idx, "dot");
			}

			if (!getenv("SKIP_VALIDATION")) {
				if (ws.rank == 0) fprintf(stderr, "Validating BFS %d\n", bfs_root_idx);
				b32 validation_passed = oned_graph_bfs_validate(&ws, &validation_arena, &g, &bfs, root);
				arena_clear(&validation_arena);
				if (!validation_passed) {
					if (ws.rank == 0) fprintf(stderr, "Validation failed for this BFS root; skipping rest.\n");
					break;
				}
			}
		}

		arena_release_gpu(&validation_arena);
		oned_graph_bfs_destroy(&bfs);
	}

	if (getenv("WRITE_DOT")) {
		tuple_graph_write_dot_files(&ws, &tg, "dot");
		oned_graph_write_dot_files(&ws, &g, "dot");
	}
	oned_graph_free(&g);

	worker_finalize(&ws);
}
