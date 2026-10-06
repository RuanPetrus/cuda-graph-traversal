#include "worker.h"
#include "graph_generation.h"
#include "traversal.h"
#include "visualization.h"

#include <math.h>
#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#define GPU_MEMORY_SIZE 0
#define BFS_ROOT_COUNT 64

enum {s_minimum, s_firstquartile, s_median, s_thirdquartile, s_maximum, s_mean, s_std, s_LAST};

static i32 compare_doubles(const void *a, const void *b) {
	f64 aa = *(const f64 *)a;
	f64 bb = *(const f64 *)b;
	return (aa < bb) ? -1 : (aa == bb) ? 0 : 1;
}

static void get_statistics(Worker_State *ws, const f64 *x, i32 n, volatile f64 r[s_LAST]) {
	f64 mean = 0.0;
	for (i32 i = 0; i < n; ++i) mean += x[i];
	mean /= n;
	r[s_mean] = mean;

	f64 variance = 0.0;
	for (i32 i = 0; i < n; ++i) variance += (x[i] - mean) * (x[i] - mean);
	variance /= n - 1;
	r[s_std] = sqrt(variance);

	f64 *sorted = arena_push_array(&ws->cpu_arena, n, f64);
	memcpy(sorted, x, (u64)n * sizeof(f64));
	qsort(sorted, n, sizeof(f64), compare_doubles);
	r[s_minimum] = sorted[0];
	r[s_firstquartile] = (sorted[(n - 1) / 4] + sorted[n / 4]) * 0.5;
	r[s_median] = (sorted[(n - 1) / 2] + sorted[n / 2]) * 0.5;
	r[s_thirdquartile] = (sorted[n - 1 - (n - 1) / 4] + sorted[n - 1 - n / 4]) * 0.5;
	r[s_maximum] = sorted[n - 1];
}

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

__global__ static void bfs_edge_count_kernel(const u64 *rowstarts, const u64 *column, const i64 *pred, u64 nlocalverts, u64 *edge_count, i32 rank, i32 npes) {
	u64 tid = (u64)blockIdx.x * blockDim.x + threadIdx.x;
	u64 stride = (u64)blockDim.x * gridDim.x;

	for (u64 local_vertex = tid; local_vertex < nlocalverts; local_vertex += stride) {
		if (pred[local_vertex] == -1) continue;
		u64 global_vertex = local_vertex * (u64)npes + (u64)rank;
		for (u64 edge_idx = rowstarts[local_vertex]; edge_idx < rowstarts[local_vertex + 1]; ++edge_idx) {
			if (column[edge_idx] <= global_vertex) atomicAdd((unsigned long long *)edge_count, 1ULL);
		}
	}
}

static u64 oned_graph_edge_count(Worker_State *ws, Arena *temp_arena, const Oned_Graph *g, const i64 *pred) {
	u64 *local_edge_count_gpu = arena_push_array(temp_arena, 1, u64);
	CHECK_CUDA(cudaMemset(local_edge_count_gpu, 0, sizeof(u64)));

	i32 threads_per_block = 256;
	u64 blocks_needed = INT_CEIL(g->nlocalverts, (u64)threads_per_block);
	i32 blocks = blocks_needed > 65535 ? 65535 : (i32)blocks_needed;
	if (blocks == 0) blocks = 1;
	bfs_edge_count_kernel<<<blocks, threads_per_block>>>(g->rowstarts, g->column, pred, g->nlocalverts, local_edge_count_gpu, ws->rank, ws->rank_size);
	CHECK_CUDA(cudaGetLastError());
	CHECK_CUDA(cudaDeviceSynchronize());

	u64 local_edge_count = 0;
	u64 global_edge_count = 0;
	CHECK_CUDA(cudaMemcpy(&local_edge_count, local_edge_count_gpu, sizeof(local_edge_count), cudaMemcpyDeviceToHost));
	MPI_Allreduce(&local_edge_count, &global_edge_count, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
	return global_edge_count;
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
	worker_kernel_launch(ws, (const void *)bfs_validate_edges_kernel, edge_args, nlocalverts, 256, "BFS validation edges");
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
	double make_graph_start = MPI_Wtime();
	{
		i64 edge_per_rank_count = INT_CEIL(tg.nglobaledges, ws.rank_size);
		i64 start_edge_index = MIN(edge_per_rank_count * ws.rank, tg.nglobaledges);
		i64 end_edge_index = MIN(start_edge_index + edge_per_rank_count, tg.nglobaledges);
		generate_kronecker_range(&ws, seed, SCALE, start_edge_index, end_edge_index, &tg);
		tuple_graph_dump(&tg, start_edge_index); // Use to check against old implementation
	}
	double make_graph_stop = MPI_Wtime();

	double data_struct_start = MPI_Wtime();
	Oned_Graph g = oned_graph_from_tuple_graph(&ws, &tg, (u64)1 << SCALE);
	double data_struct_stop = MPI_Wtime();

	u64 bfs_roots[BFS_ROOT_COUNT];
	i32 bfs_root_count = generate_bfs_roots(&ws, &g, seed1, seed2, bfs_roots, BFS_ROOT_COUNT);

	if (!getenv("SKIP_BFS") && bfs_root_count > 0) {
		Bfs_State bfs = oned_graph_bfs_create(&ws, &g);
		Arena validation_arena = arena_from_arena(&ws.gpu_arena, bfs.max_nlocalverts * sizeof(i32) + sizeof(u64) + KILOBYTE(4));
		f64 *bfs_times = arena_push_array(&ws.cpu_arena, bfs_root_count, f64);
		f64 *validate_times = arena_push_array(&ws.cpu_arena, bfs_root_count, f64);
		f64 *edge_counts = arena_push_array(&ws.cpu_arena, bfs_root_count, f64);
		i32 completed_bfs_count = 0;
		b32 validation_passed_all = true;

		oned_graph_bfs_clear(&ws, &bfs);
		oned_graph_bfs_run(&ws, &g, &bfs, bfs_roots[0]); // Warm-up, like the original Graph500 runner.

		for (i32 bfs_root_idx = 0; bfs_root_idx < bfs_root_count; ++bfs_root_idx) {
			u64 root = bfs_roots[bfs_root_idx];

			oned_graph_bfs_clear(&ws, &bfs);
			double bfs_start = MPI_Wtime();
			oned_graph_bfs_run(&ws, &g, &bfs, root);
			double bfs_stop = MPI_Wtime();
			bfs_times[bfs_root_idx] = bfs_stop - bfs_start;
			edge_counts[bfs_root_idx] = (f64)oned_graph_edge_count(&ws, &validation_arena, &g, bfs.pred);
			arena_clear(&validation_arena);
			if (getenv("WRITE_DOT")) {
				bfs_compute_dist_from_pred(&ws, &g, &bfs, root);
				bfs_write_dot_files_for_root(&ws, &g, &bfs, root, bfs_root_idx, "dot");
			}

			if (!getenv("SKIP_VALIDATION")) {
				double validate_start = MPI_Wtime();
				b32 validation_passed = oned_graph_bfs_validate(&ws, &validation_arena, &g, &bfs, root);
				double validate_stop = MPI_Wtime();
				validate_times[bfs_root_idx] = validate_stop - validate_start;
				arena_clear(&validation_arena);
				if (!validation_passed) {
					if (ws.rank == 0) fprintf(stderr, "Validation failed for this BFS root; skipping rest.\n");
					validation_passed_all = false;
					break;
				}
			} else {
				validate_times[bfs_root_idx] = -1.0;
			}
			completed_bfs_count = bfs_root_idx + 1;
		}

		if (ws.rank == 0 && validation_passed_all && completed_bfs_count > 0) {
			volatile f64 stats[s_LAST];
			fprintf(stdout, "SCALE:                          %d\n", SCALE);
			fprintf(stdout, "edgefactor:                     %d\n", edgefactor);
			fprintf(stdout, "NBFS:                           %d\n", completed_bfs_count);
			fprintf(stdout, "graph_generation:               %g\n", make_graph_stop - make_graph_start);
			fprintf(stdout, "num_mpi_processes:              %d\n", ws.rank_size);
			fprintf(stdout, "construction_time:              %g\n", data_struct_stop - data_struct_start);

			get_statistics(&ws, bfs_times, completed_bfs_count, stats);
			fprintf(stdout, "bfs  min_time:                  %g\n", stats[s_minimum]);
			fprintf(stdout, "bfs  firstquartile_time:        %g\n", stats[s_firstquartile]);
			fprintf(stdout, "bfs  median_time:               %g\n", stats[s_median]);
			fprintf(stdout, "bfs  thirdquartile_time:        %g\n", stats[s_thirdquartile]);
			fprintf(stdout, "bfs  max_time:                  %g\n", stats[s_maximum]);
			fprintf(stdout, "bfs  mean_time:                 %g\n", stats[s_mean]);
			fprintf(stdout, "bfs  stddev_time:               %g\n", stats[s_std]);

			get_statistics(&ws, edge_counts, completed_bfs_count, stats);
			fprintf(stdout, "min_nedge:                      %.11g\n", stats[s_minimum]);
			fprintf(stdout, "firstquartile_nedge:            %.11g\n", stats[s_firstquartile]);
			fprintf(stdout, "median_nedge:                   %.11g\n", stats[s_median]);
			fprintf(stdout, "thirdquartile_nedge:            %.11g\n", stats[s_thirdquartile]);
			fprintf(stdout, "max_nedge:                      %.11g\n", stats[s_maximum]);
			fprintf(stdout, "mean_nedge:                     %.11g\n", stats[s_mean]);
			fprintf(stdout, "stddev_nedge:                   %.11g\n", stats[s_std]);

			f64 *secs_per_edge = arena_push_array(&ws.cpu_arena, completed_bfs_count, f64);
			for (i32 i = 0; i < completed_bfs_count; ++i) secs_per_edge[i] = bfs_times[i] / edge_counts[i];
			get_statistics(&ws, secs_per_edge, completed_bfs_count, stats);
			fprintf(stdout, "bfs  min_TEPS:                  %g\n", 1.0 / stats[s_maximum]);
			fprintf(stdout, "bfs  firstquartile_TEPS:        %g\n", 1.0 / stats[s_thirdquartile]);
			fprintf(stdout, "bfs  median_TEPS:               %g\n", 1.0 / stats[s_median]);
			fprintf(stdout, "bfs  thirdquartile_TEPS:        %g\n", 1.0 / stats[s_firstquartile]);
			fprintf(stdout, "bfs  max_TEPS:                  %g\n", 1.0 / stats[s_minimum]);
			fprintf(stdout, "bfs  harmonic_mean_TEPS:     !  %g\n", 1.0 / stats[s_mean]);
			fprintf(stdout, "bfs  harmonic_stddev_TEPS:      %g\n", stats[s_std] / (stats[s_mean] * stats[s_mean] * sqrt(completed_bfs_count - 1)));

			if (!getenv("SKIP_VALIDATION")) {
				get_statistics(&ws, validate_times, completed_bfs_count, stats);
				fprintf(stdout, "bfs  min_validate:              %g\n", stats[s_minimum]);
				fprintf(stdout, "bfs  firstquartile_validate:    %g\n", stats[s_firstquartile]);
				fprintf(stdout, "bfs  median_validate:           %g\n", stats[s_median]);
				fprintf(stdout, "bfs  thirdquartile_validate:    %g\n", stats[s_thirdquartile]);
				fprintf(stdout, "bfs  max_validate:              %g\n", stats[s_maximum]);
				fprintf(stdout, "bfs  mean_validate:             %g\n", stats[s_mean]);
				fprintf(stdout, "bfs  stddev_validate:           %g\n", stats[s_std]);
			}
		} else if (ws.rank == 0 && !validation_passed_all) {
			fprintf(stdout, "No results printed for invalid run.\n");
		}

		oned_graph_bfs_destroy(&bfs);
	}

	if (!getenv("SKIP_SSSP") && bfs_root_count > 0) {
		Sssp_State sssp = oned_graph_sssp_create(&ws, &g);
		Arena count_arena = arena_from_arena(&ws.gpu_arena, sizeof(u64) + KILOBYTE(4));
		f64 *sssp_times = arena_push_array(&ws.cpu_arena, bfs_root_count, f64);
		f64 *edge_counts = arena_push_array(&ws.cpu_arena, bfs_root_count, f64);

		oned_graph_sssp_clear(&ws, &sssp);
		oned_graph_sssp_run(&ws, &g, &sssp, bfs_roots[0]);

		for (i32 sssp_root_idx = 0; sssp_root_idx < bfs_root_count; ++sssp_root_idx) {
			u64 root = bfs_roots[sssp_root_idx];
			oned_graph_sssp_clear(&ws, &sssp);
			double sssp_start = MPI_Wtime();
			oned_graph_sssp_run(&ws, &g, &sssp, root);
			double sssp_stop = MPI_Wtime();
			sssp_times[sssp_root_idx] = sssp_stop - sssp_start;
			edge_counts[sssp_root_idx] = (f64)oned_graph_edge_count(&ws, &count_arena, &g, sssp.pred);
			arena_clear(&count_arena);
		}

		if (ws.rank == 0) {
			volatile f64 stats[s_LAST];
			get_statistics(&ws, sssp_times, bfs_root_count, stats);
			fprintf(stdout, "sssp min_time:                  %g\n", stats[s_minimum]);
			fprintf(stdout, "sssp firstquartile_time:        %g\n", stats[s_firstquartile]);
			fprintf(stdout, "sssp median_time:               %g\n", stats[s_median]);
			fprintf(stdout, "sssp thirdquartile_time:        %g\n", stats[s_thirdquartile]);
			fprintf(stdout, "sssp max_time:                  %g\n", stats[s_maximum]);
			fprintf(stdout, "sssp mean_time:                 %g\n", stats[s_mean]);
			fprintf(stdout, "sssp stddev_time:               %g\n", stats[s_std]);

			for (i32 i = 0; i < bfs_root_count; ++i) edge_counts[i] = sssp_times[i] / edge_counts[i];
			get_statistics(&ws, edge_counts, bfs_root_count, stats);
			fprintf(stdout, "sssp min_TEPS:                  %g\n", 1.0 / stats[s_maximum]);
			fprintf(stdout, "sssp median_TEPS:               %g\n", 1.0 / stats[s_median]);
			fprintf(stdout, "sssp max_TEPS:                  %g\n", 1.0 / stats[s_minimum]);
		}

		oned_graph_sssp_destroy(&sssp);
	}

	if (getenv("WRITE_DOT")) {
		tuple_graph_write_dot_files(&ws, &tg, "dot");
		oned_graph_write_dot_files(&ws, &g, "dot");
	}
	oned_graph_free(&g);

	worker_finalize(&ws);
}
