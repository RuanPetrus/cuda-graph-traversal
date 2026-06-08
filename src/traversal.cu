//Stub for custom BFS implementations

#include "common.h"
#ifdef __cplusplus
extern "C" {
#endif
#include "csr_reference.h"
#ifdef __cplusplus
}
#endif
#include "bitmap_reference.h"
#include <stdint.h>
#include <inttypes.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <limits.h>
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>

//VISITED bitmap parameters
unsigned long *visited;
int64_t visited_size;

int64_t *pred_glob;
extern int64_t *column;
unsigned int *rowstarts;
oned_csr_graph g;

unsigned int *nvshmem_rowstarts;
char *nvshmem_column;
int64_t *nvshmem_pred;
int *nvshmem_frontier[2];
int *nvshmem_frontier_count[2];
size_t max_nvshmem_rowstarts_count;
size_t max_nvshmem_column_bytes;
size_t max_nvshmem_nlocalverts;
#ifdef SSSP
float *nvshmem_weights;
unsigned int *nvshmem_dist_bits;
int *nvshmem_sssp_frontier[2];
int *nvshmem_sssp_frontier_count[2];
int *nvshmem_sssp_visited;
int *nvshmem_sssp_pred_lock;
size_t max_nvshmem_nlocaledges;
#endif
static int nvshmem_ready;

#define CHECK_CUDA(call) do { \
	cudaError_t err_ = (call); \
	if (err_ != cudaSuccess) { \
		fprintf(stderr, "Rank %d: %s:%d: %s failed: %s\n", rank, __FILE__, __LINE__, #call, cudaGetErrorString(err_)); \
		MPI_Abort(MPI_COMM_WORLD, 1); \
	} \
} while (0)

#define NVSHMEM_MALLOC_OR_ABORT(ptr, bytes) do { \
	(ptr) = (decltype(+ptr))nvshmem_malloc(bytes); \
	if ((bytes) != 0 && (ptr) == NULL) { \
		fprintf(stderr, "Rank %d: %s:%d: nvshmem_malloc failed for %s (%zu bytes)\n", rank, __FILE__, __LINE__, #ptr, (size_t)(bytes)); \
		MPI_Abort(MPI_COMM_WORLD, 1); \
	} \
} while (0)

#define CHECK_NVSHMEM_LAUNCH(call) do { \
	int err_ = (call); \
	if (err_ != 0) { \
		fprintf(stderr, "Rank %d: %s:%d: %s failed with error %d\n", rank, __FILE__, __LINE__, #call, err_); \
		MPI_Abort(MPI_COMM_WORLD, 1); \
	} \
} while (0)

__device__ static int vertex_owner_device(int64_t v, int npes, int lg_npes) {
#ifdef SIZE_MUST_BE_A_POWER_OF_TWO
	return (int)(v & ((1 << lg_npes) - 1));
#else
	return (int)(v % npes);
#endif
}

__device__ static int vertex_local_device(int64_t v, int npes, int lg_npes) {
#ifdef SIZE_MUST_BE_A_POWER_OF_TWO
	return (int)(v >> lg_npes);
#else
	return (int)(v / npes);
#endif
}

__device__ static int64_t vertex_to_global_device(int pe, int local, int npes, int lg_npes) {
#ifdef SIZE_MUST_BE_A_POWER_OF_TWO
	return ((int64_t)local << lg_npes) + pe;
#else
	return ((int64_t)local * npes) + pe;
#endif
}

__device__ static int64_t get_column_device(const char *column_data, size_t edge_idx) {
	uint64_t value = 0;
	const unsigned char *src = (const unsigned char*)column_data + BYTES_PER_VERTEX * edge_idx;
	for (int i = 0; i < BYTES_PER_VERTEX; ++i) value |= ((uint64_t)src[i]) << (8 * i);
	return (int64_t)value;
}

__global__ static void bfs_root_init_kernel(int64_t root, int64_t *pred, int *frontier,
		int *frontier_count, int *next_frontier_count, int npes, int lg_npes) {
	if (blockIdx.x != 0 || threadIdx.x != 0) return;

	*frontier_count = 0;
	*next_frontier_count = 0;
	int my_pe = nvshmem_my_pe();
	if (vertex_owner_device(root, npes, lg_npes) == my_pe) {
		int root_local = vertex_local_device(root, npes, lg_npes);
		pred[root_local] = root;
		frontier[0] = root_local;
		*frontier_count = 1;
	}
}

__global__ static void bfs_expand_kernel(const unsigned int *rowstarts, const char *column_data,
		int64_t *pred, const int *frontier, int *next_frontier,
		const int *frontier_count, int *next_frontier_count, int npes, int lg_npes) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	int count = *frontier_count;
	int my_pe = nvshmem_my_pe();

	for (int idx = tid; idx < count; idx += stride) {
		int src_local = frontier[idx];
		int64_t src_global = vertex_to_global_device(my_pe, src_local, npes, lg_npes);
		for (unsigned int edge = rowstarts[src_local]; edge < rowstarts[src_local + 1]; ++edge) {
			int64_t dst_global = get_column_device(column_data, edge);
			int dst_pe = vertex_owner_device(dst_global, npes, lg_npes);
			int dst_local = vertex_local_device(dst_global, npes, lg_npes);
			int64_t old = nvshmem_int64_atomic_compare_swap(&pred[dst_local], -1, src_global, dst_pe);
			if (old == -1) {
				int pos = nvshmem_int_atomic_fetch_add(next_frontier_count, 1, dst_pe);
				nvshmem_int_p(&next_frontier[pos], dst_local, dst_pe);
			}
		}
	}
	nvshmem_quiet();
}

static void init_cuda_device(void) {
	int device_count = 0;
	CHECK_CUDA(cudaGetDeviceCount(&device_count));
	if (device_count <= 0) {
		fprintf(stderr, "Rank %d: no CUDA devices available\n", rank);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	CHECK_CUDA(cudaSetDevice(rank % device_count));
}

static void init_nvshmem(void) {
	if (nvshmem_ready) return;

	init_cuda_device();

	nvshmemx_init_attr_t attr;
	memset(&attr, 0, sizeof(attr));
	MPI_Comm mpi_comm = MPI_COMM_WORLD;
	attr.mpi_comm = &mpi_comm;
	nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
	nvshmem_ready = 1;
}

//user should provide this function which would be called once to do kernel 1: graph convert
void make_graph_data_structure(const tuple_graph* const tg) {
	//graph conversion, can be changed by user by replacing oned_csr.{c,h} with new graph format 
	convert_graph_to_oned_csr(tg, &g);

	column=g.column;
	rowstarts=g.rowstarts;
	visited_size = (g.nlocalverts + ulong_bits - 1) / ulong_bits;
	visited = (unsigned long*) xmalloc(visited_size*sizeof(unsigned long));

	init_nvshmem();

	size_t local_rowstarts_count = g.nlocalverts + 1;
	size_t local_column_bytes = BYTES_PER_VERTEX * g.nlocaledges;
	size_t local_nlocalverts = g.nlocalverts;
	size_t local_nlocaledges = g.nlocaledges;
	MPI_Allreduce(&local_rowstarts_count, &max_nvshmem_rowstarts_count, 1, MPI_UNSIGNED_LONG, MPI_MAX, MPI_COMM_WORLD);
	MPI_Allreduce(&local_column_bytes, &max_nvshmem_column_bytes, 1, MPI_UNSIGNED_LONG, MPI_MAX, MPI_COMM_WORLD);
	MPI_Allreduce(&local_nlocalverts, &max_nvshmem_nlocalverts, 1, MPI_UNSIGNED_LONG, MPI_MAX, MPI_COMM_WORLD);
#ifdef SSSP
	MPI_Allreduce(&local_nlocaledges, &max_nvshmem_nlocaledges, 1, MPI_UNSIGNED_LONG, MPI_MAX, MPI_COMM_WORLD);
#endif

	NVSHMEM_MALLOC_OR_ABORT(nvshmem_rowstarts, max_nvshmem_rowstarts_count * sizeof(unsigned int));
	NVSHMEM_MALLOC_OR_ABORT(nvshmem_column, max_nvshmem_column_bytes);
	NVSHMEM_MALLOC_OR_ABORT(nvshmem_pred, max_nvshmem_nlocalverts * sizeof(int64_t));
	NVSHMEM_MALLOC_OR_ABORT(nvshmem_frontier[0], max_nvshmem_nlocalverts * sizeof(int));
	NVSHMEM_MALLOC_OR_ABORT(nvshmem_frontier[1], max_nvshmem_nlocalverts * sizeof(int));
	NVSHMEM_MALLOC_OR_ABORT(nvshmem_frontier_count[0], sizeof(int));
	NVSHMEM_MALLOC_OR_ABORT(nvshmem_frontier_count[1], sizeof(int));
#ifdef SSSP
	NVSHMEM_MALLOC_OR_ABORT(nvshmem_weights, max_nvshmem_nlocaledges * sizeof(float));
	NVSHMEM_MALLOC_OR_ABORT(nvshmem_dist_bits, max_nvshmem_nlocalverts * sizeof(unsigned int));
	NVSHMEM_MALLOC_OR_ABORT(nvshmem_sssp_frontier[0], max_nvshmem_nlocalverts * sizeof(int));
	NVSHMEM_MALLOC_OR_ABORT(nvshmem_sssp_frontier[1], max_nvshmem_nlocalverts * sizeof(int));
	NVSHMEM_MALLOC_OR_ABORT(nvshmem_sssp_frontier_count[0], sizeof(int));
	NVSHMEM_MALLOC_OR_ABORT(nvshmem_sssp_frontier_count[1], sizeof(int));
	NVSHMEM_MALLOC_OR_ABORT(nvshmem_sssp_visited, max_nvshmem_nlocalverts * sizeof(int));
	NVSHMEM_MALLOC_OR_ABORT(nvshmem_sssp_pred_lock, max_nvshmem_nlocalverts * sizeof(int));
#endif

	CHECK_CUDA(cudaMemset(nvshmem_rowstarts, 0, max_nvshmem_rowstarts_count * sizeof(unsigned int)));
	CHECK_CUDA(cudaMemset(nvshmem_column, 0, max_nvshmem_column_bytes));
	CHECK_CUDA(cudaMemset(nvshmem_pred, 0xff, max_nvshmem_nlocalverts * sizeof(int64_t)));
	CHECK_CUDA(cudaMemset(nvshmem_frontier[0], 0, max_nvshmem_nlocalverts * sizeof(int)));
	CHECK_CUDA(cudaMemset(nvshmem_frontier[1], 0, max_nvshmem_nlocalverts * sizeof(int)));
	CHECK_CUDA(cudaMemset(nvshmem_frontier_count[0], 0, sizeof(int)));
	CHECK_CUDA(cudaMemset(nvshmem_frontier_count[1], 0, sizeof(int)));
#ifdef SSSP
	CHECK_CUDA(cudaMemset(nvshmem_weights, 0, max_nvshmem_nlocaledges * sizeof(float)));
	CHECK_CUDA(cudaMemset(nvshmem_dist_bits, 0x7f, max_nvshmem_nlocalverts * sizeof(unsigned int)));
	CHECK_CUDA(cudaMemset(nvshmem_sssp_frontier[0], 0, max_nvshmem_nlocalverts * sizeof(int)));
	CHECK_CUDA(cudaMemset(nvshmem_sssp_frontier[1], 0, max_nvshmem_nlocalverts * sizeof(int)));
	CHECK_CUDA(cudaMemset(nvshmem_sssp_frontier_count[0], 0, sizeof(int)));
	CHECK_CUDA(cudaMemset(nvshmem_sssp_frontier_count[1], 0, sizeof(int)));
	CHECK_CUDA(cudaMemset(nvshmem_sssp_visited, 0, max_nvshmem_nlocalverts * sizeof(int)));
	CHECK_CUDA(cudaMemset(nvshmem_sssp_pred_lock, 0, max_nvshmem_nlocalverts * sizeof(int)));
#endif
	CHECK_CUDA(cudaMemcpy(nvshmem_rowstarts, g.rowstarts, local_rowstarts_count * sizeof(unsigned int), cudaMemcpyHostToDevice));
	CHECK_CUDA(cudaMemcpy(nvshmem_column, g.column, local_column_bytes, cudaMemcpyHostToDevice));
#ifdef SSSP
	CHECK_CUDA(cudaMemcpy(nvshmem_weights, g.weights, local_nlocaledges * sizeof(float), cudaMemcpyHostToDevice));
#endif
	CHECK_CUDA(cudaDeviceSynchronize());
	nvshmem_barrier_all();
}

//user should provide this function which would be called several times to do kernel 2: breadth first search
//pred[] should be root for root, -1 for unrechable vertices
//prior to calling run_bfs pred is set to -1 by calling clean_pred
void run_bfs(int64_t root, int64_t* pred) {
	pred_glob=pred;
	int current = 0;
	int next = 1;
	int npes = size;
	int lg_npes = 0;
#ifdef SIZE_MUST_BE_A_POWER_OF_TWO
	lg_npes = lgsize;
#endif
	dim3 init_grid(1);
	dim3 init_block(1);
	void *init_args[] = {&root, &nvshmem_pred, &nvshmem_frontier[current],
		&nvshmem_frontier_count[current], &nvshmem_frontier_count[next], &npes, &lg_npes};
	CHECK_NVSHMEM_LAUNCH(nvshmemx_collective_launch((const void*)bfs_root_init_kernel,
		init_grid, init_block, init_args, 0, 0));
	CHECK_CUDA(cudaDeviceSynchronize());
	nvshmem_barrier_all();

	int local_count = 0;
	int64_t local_count_64 = 0;
	int64_t global_count = 0;
	CHECK_CUDA(cudaMemcpy(&local_count, nvshmem_frontier_count[current], sizeof(int), cudaMemcpyDeviceToHost));
	local_count_64 = local_count;
	MPI_Allreduce(&local_count_64, &global_count, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);

	dim3 expand_block(256);
	dim3 expand_grid(128);
	while (global_count > 0) {
		CHECK_CUDA(cudaMemset(nvshmem_frontier_count[next], 0, sizeof(int)));
		CHECK_CUDA(cudaDeviceSynchronize());
		nvshmem_barrier_all();

		void *expand_args[] = {&nvshmem_rowstarts, &nvshmem_column, &nvshmem_pred,
			&nvshmem_frontier[current], &nvshmem_frontier[next],
			&nvshmem_frontier_count[current], &nvshmem_frontier_count[next], &npes, &lg_npes};
		CHECK_NVSHMEM_LAUNCH(nvshmemx_collective_launch((const void*)bfs_expand_kernel,
			expand_grid, expand_block, expand_args, 0, 0));
		CHECK_CUDA(cudaDeviceSynchronize());
		nvshmem_barrier_all();

		CHECK_CUDA(cudaMemcpy(&local_count, nvshmem_frontier_count[next], sizeof(int), cudaMemcpyDeviceToHost));
		local_count_64 = local_count;
		MPI_Allreduce(&local_count_64, &global_count, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);

		int tmp = current;
		current = next;
		next = tmp;
	}

	CHECK_CUDA(cudaMemcpy(pred, nvshmem_pred, g.nlocalverts * sizeof(int64_t), cudaMemcpyDeviceToHost));
	CHECK_CUDA(cudaDeviceSynchronize());
}

//we need edge count to calculate teps. Validation will check if this count is correct
//user should change this function if another format (not standart CRS) used
void get_edge_count_for_teps(int64_t* edge_visit_count) {
	long i,j;
	int64_t edge_count=0;
	for(i=0;i<g.nlocalverts;i++)
		if(pred_glob[i]!=-1) {
			for(j=g.rowstarts[i];j<g.rowstarts[i+1];j++)
				if(COLUMN(j)<=VERTEX_TO_GLOBAL(rank,i))
					edge_count++;
		}

	MPI_Allreduce(&edge_count, edge_visit_count, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
}

//user provided function to initialize predecessor array to whatevere value user needs
void clean_pred(int64_t* pred) {
	int i;
	for(i=0;i<g.nlocalverts;i++) pred[i]=-1;
	if (nvshmem_pred != NULL) {
		CHECK_CUDA(cudaMemset(nvshmem_pred, 0xff, g.nlocalverts * sizeof(int64_t)));
		CHECK_CUDA(cudaDeviceSynchronize());
		nvshmem_barrier_all();
	}
}

//user provided function to be called once graph is no longer needed
void free_graph_data_structure(void) {
	if (nvshmem_ready) {
		nvshmem_barrier_all();
#ifdef SSSP
		if (nvshmem_sssp_pred_lock != NULL) {
			nvshmem_free(nvshmem_sssp_pred_lock);
			nvshmem_sssp_pred_lock = NULL;
		}
		if (nvshmem_sssp_visited != NULL) {
			nvshmem_free(nvshmem_sssp_visited);
			nvshmem_sssp_visited = NULL;
		}
		if (nvshmem_sssp_frontier_count[1] != NULL) {
			nvshmem_free(nvshmem_sssp_frontier_count[1]);
			nvshmem_sssp_frontier_count[1] = NULL;
		}
		if (nvshmem_sssp_frontier_count[0] != NULL) {
			nvshmem_free(nvshmem_sssp_frontier_count[0]);
			nvshmem_sssp_frontier_count[0] = NULL;
		}
		if (nvshmem_sssp_frontier[1] != NULL) {
			nvshmem_free(nvshmem_sssp_frontier[1]);
			nvshmem_sssp_frontier[1] = NULL;
		}
		if (nvshmem_sssp_frontier[0] != NULL) {
			nvshmem_free(nvshmem_sssp_frontier[0]);
			nvshmem_sssp_frontier[0] = NULL;
		}
		if (nvshmem_dist_bits != NULL) {
			nvshmem_free(nvshmem_dist_bits);
			nvshmem_dist_bits = NULL;
		}
		if (nvshmem_weights != NULL) {
			nvshmem_free(nvshmem_weights);
			nvshmem_weights = NULL;
		}
#endif
		if (nvshmem_frontier_count[1] != NULL) {
			nvshmem_free(nvshmem_frontier_count[1]);
			nvshmem_frontier_count[1] = NULL;
		}
		if (nvshmem_frontier_count[0] != NULL) {
			nvshmem_free(nvshmem_frontier_count[0]);
			nvshmem_frontier_count[0] = NULL;
		}
		if (nvshmem_frontier[1] != NULL) {
			nvshmem_free(nvshmem_frontier[1]);
			nvshmem_frontier[1] = NULL;
		}
		if (nvshmem_frontier[0] != NULL) {
			nvshmem_free(nvshmem_frontier[0]);
			nvshmem_frontier[0] = NULL;
		}
		if (nvshmem_pred != NULL) {
			nvshmem_free(nvshmem_pred);
			nvshmem_pred = NULL;
		}
		if (nvshmem_column != NULL) {
			nvshmem_free(nvshmem_column);
			nvshmem_column = NULL;
		}
		if (nvshmem_rowstarts != NULL) {
			nvshmem_free(nvshmem_rowstarts);
			nvshmem_rowstarts = NULL;
		}
		nvshmem_finalize();
		nvshmem_ready = 0;
	}
	free_oned_csr_graph(&g);
	free(visited);
}

//user should change is function if distribution(and counts) of vertices is changed
size_t get_nlocalverts_for_pred(void) {
	return g.nlocalverts;
}

#ifdef SSSP

static const unsigned int SSSP_INF_BITS = 0x7f800000U;

__device__ static void sssp_relax(unsigned int *dist_bits, int64_t *pred,
		int *next_frontier, int *next_frontier_count, int *visited_flags,
		int *pred_lock,
		int64_t dst_global, float candidate_dist, int src_local,
		float bucket_max, int light_phase, int npes, int lg_npes) {
	int my_pe = nvshmem_my_pe();
	int dst_pe = vertex_owner_device(dst_global, npes, lg_npes);
	int dst_local = vertex_local_device(dst_global, npes, lg_npes);
	int64_t src_global = vertex_to_global_device(my_pe, src_local, npes, lg_npes);
	unsigned int candidate_bits = __float_as_uint(candidate_dist);
	unsigned int old_bits = nvshmem_uint_g(&dist_bits[dst_local], dst_pe);

	while (candidate_bits < old_bits) {
		unsigned int prev = nvshmem_uint_atomic_compare_swap(&dist_bits[dst_local], old_bits, candidate_bits, dst_pe);
		if (prev == old_bits) {
			while (nvshmem_int_atomic_compare_swap(&pred_lock[dst_local], 0, 1, dst_pe) != 0) {}
			if (nvshmem_uint_g(&dist_bits[dst_local], dst_pe) == candidate_bits) {
				nvshmem_int64_p(&pred[dst_local], src_global, dst_pe);
			}
			nvshmem_int_p(&pred_lock[dst_local], 0, dst_pe);
			if (light_phase && candidate_dist < bucket_max) {
				int was_seen = nvshmem_int_atomic_compare_swap(&visited_flags[dst_local], 0, 1, dst_pe);
				if (was_seen == 0) {
					int pos = nvshmem_int_atomic_fetch_add(next_frontier_count, 1, dst_pe);
					nvshmem_int_p(&next_frontier[pos], dst_local, dst_pe);
				}
			}
			break;
		}
		old_bits = prev;
	}
}

__global__ static void sssp_clean_dist_kernel(unsigned int *dist_bits, size_t nlocalverts) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (size_t i = tid; i < nlocalverts; i += stride) dist_bits[i] = SSSP_INF_BITS;
}

__global__ static void sssp_root_init_kernel(int64_t root, unsigned int *dist_bits, int64_t *pred,
		int *frontier, int *frontier_count, int *next_frontier_count, int npes, int lg_npes) {
	if (blockIdx.x != 0 || threadIdx.x != 0) return;
	*frontier_count = 0;
	*next_frontier_count = 0;
	int my_pe = nvshmem_my_pe();
	if (vertex_owner_device(root, npes, lg_npes) == my_pe) {
		int root_local = vertex_local_device(root, npes, lg_npes);
		dist_bits[root_local] = __float_as_uint(0.0f);
		pred[root_local] = root;
		frontier[0] = root_local;
		*frontier_count = 1;
	}
}

__global__ static void sssp_clear_light_kernel(int *visited_flags, int *next_frontier_count, size_t nlocalverts) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	if (tid == 0) *next_frontier_count = 0;
	for (size_t i = tid; i < nlocalverts; i += stride) visited_flags[i] = 0;
}

__global__ static void sssp_light_relax_kernel(const unsigned int *rowstarts, const char *column_data,
		const float *weights, unsigned int *dist_bits, int64_t *pred,
		const int *frontier, int *next_frontier, const int *frontier_count,
		int *next_frontier_count, int *visited_flags, int *pred_lock, float delta, float bucket_max,
		int npes, int lg_npes) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	int count = *frontier_count;
	for (int idx = tid; idx < count; idx += stride) {
		int src_local = frontier[idx];
		float src_dist = __uint_as_float(dist_bits[src_local]);
		for (unsigned int edge = rowstarts[src_local]; edge < rowstarts[src_local + 1]; ++edge) {
			float edge_weight = weights[edge];
			if (edge_weight < delta) {
				sssp_relax(dist_bits, pred, next_frontier, next_frontier_count, visited_flags,
					pred_lock,
					get_column_device(column_data, edge), src_dist + edge_weight, src_local,
					bucket_max, 1, npes, lg_npes);
			}
		}
	}
	nvshmem_quiet();
}

__global__ static void sssp_heavy_relax_kernel(const unsigned int *rowstarts, const char *column_data,
		const float *weights, unsigned int *dist_bits, int64_t *pred,
		int *pred_lock, float delta, float bucket_min, float bucket_max, size_t nlocalverts, int npes, int lg_npes) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (size_t src_local = tid; src_local < nlocalverts; src_local += stride) {
		float src_dist = __uint_as_float(dist_bits[src_local]);
		if (src_dist >= bucket_min && src_dist < bucket_max) {
			for (unsigned int edge = rowstarts[src_local]; edge < rowstarts[src_local + 1]; ++edge) {
				float edge_weight = weights[edge];
				if (edge_weight >= delta) {
					sssp_relax(dist_bits, pred, NULL, NULL, NULL, pred_lock, get_column_device(column_data, edge),
						src_dist + edge_weight, (int)src_local, bucket_max, 0, npes, lg_npes);
				}
			}
		}
	}
	nvshmem_quiet();
}

__global__ static void sssp_bucket_scan_kernel(const unsigned int *dist_bits, int *frontier,
		int *frontier_count, unsigned long long *pending_count, float bucket_min,
		float bucket_max, size_t nlocalverts) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (size_t i = tid; i < nlocalverts; i += stride) {
		if (dist_bits[i] == SSSP_INF_BITS) continue;
		float d = __uint_as_float(dist_bits[i]);
		if (d >= bucket_min) {
			atomicAdd(pending_count, 1ULL);
			if (d < bucket_max) {
				int pos = atomicAdd(frontier_count, 1);
				frontier[pos] = (int)i;
			}
		}
	}
}

__global__ static void sssp_export_dist_kernel(const unsigned int *dist_bits, float *dist, size_t nlocalverts) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (size_t i = tid; i < nlocalverts; i += stride) {
		unsigned int bits = dist_bits[i];
		dist[i] = (bits == SSSP_INF_BITS) ? -1.0f : __uint_as_float(bits);
	}
}

static unsigned long long *get_pending_counter(void) {
	static unsigned long long *counter = NULL;
	if (counter == NULL) CHECK_CUDA(cudaMalloc(&counter, sizeof(unsigned long long)));
	return counter;
}

void run_sssp(int64_t root,int64_t* pred,float *dist) {
	pred_glob=pred;
	float delta = 0.1f;
	float bucket_min = 0.0f;
	float bucket_max = delta;
	int current = 0;
	int next = 1;
	int npes = size;
	int lg_npes = 0;
#ifdef SIZE_MUST_BE_A_POWER_OF_TWO
	lg_npes = lgsize;
#endif
	dim3 block(256);
	dim3 grid(128);
	unsigned long long *pending_counter = get_pending_counter();

	void *init_args[] = {&root, &nvshmem_dist_bits, &nvshmem_pred,
		&nvshmem_sssp_frontier[current], &nvshmem_sssp_frontier_count[current],
		&nvshmem_sssp_frontier_count[next], &npes, &lg_npes};
	CHECK_NVSHMEM_LAUNCH(nvshmemx_collective_launch((const void*)sssp_root_init_kernel,
		dim3(1), dim3(1), init_args, 0, 0));
	CHECK_CUDA(cudaDeviceSynchronize());
	nvshmem_barrier_all();

	int local_count = 0;
	int64_t local_count_64 = 0;
	int64_t global_count = 1;
	while (global_count != 0) {
		CHECK_CUDA(cudaMemcpy(&local_count, nvshmem_sssp_frontier_count[current], sizeof(int), cudaMemcpyDeviceToHost));
		local_count_64 = local_count;
		MPI_Allreduce(&local_count_64, &global_count, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);

		while (global_count != 0) {
			void *clear_args[] = {&nvshmem_sssp_visited, &nvshmem_sssp_frontier_count[next], &g.nlocalverts};
			CHECK_NVSHMEM_LAUNCH(nvshmemx_collective_launch((const void*)sssp_clear_light_kernel,
				grid, block, clear_args, 0, 0));
			CHECK_CUDA(cudaDeviceSynchronize());
			nvshmem_barrier_all();

			void *light_args[] = {&nvshmem_rowstarts, &nvshmem_column, &nvshmem_weights,
				&nvshmem_dist_bits, &nvshmem_pred, &nvshmem_sssp_frontier[current],
				&nvshmem_sssp_frontier[next], &nvshmem_sssp_frontier_count[current],
				&nvshmem_sssp_frontier_count[next], &nvshmem_sssp_visited, &nvshmem_sssp_pred_lock, &delta,
				&bucket_max, &npes, &lg_npes};
			CHECK_NVSHMEM_LAUNCH(nvshmemx_collective_launch((const void*)sssp_light_relax_kernel,
				grid, block, light_args, 0, 0));
			CHECK_CUDA(cudaDeviceSynchronize());
			nvshmem_barrier_all();

			int tmp = current;
			current = next;
			next = tmp;
			CHECK_CUDA(cudaMemcpy(&local_count, nvshmem_sssp_frontier_count[current], sizeof(int), cudaMemcpyDeviceToHost));
			local_count_64 = local_count;
			MPI_Allreduce(&local_count_64, &global_count, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
		}

		void *heavy_args[] = {&nvshmem_rowstarts, &nvshmem_column, &nvshmem_weights,
			&nvshmem_dist_bits, &nvshmem_pred, &nvshmem_sssp_pred_lock, &delta, &bucket_min, &bucket_max,
			&g.nlocalverts, &npes, &lg_npes};
		CHECK_NVSHMEM_LAUNCH(nvshmemx_collective_launch((const void*)sssp_heavy_relax_kernel,
			grid, block, heavy_args, 0, 0));
		CHECK_CUDA(cudaDeviceSynchronize());
		nvshmem_barrier_all();

		bucket_min = bucket_max;
		bucket_max += delta;
		CHECK_CUDA(cudaMemset(nvshmem_sssp_frontier_count[current], 0, sizeof(int)));
		CHECK_CUDA(cudaMemset(pending_counter, 0, sizeof(unsigned long long)));
		CHECK_CUDA(cudaDeviceSynchronize());
		void *scan_args[] = {&nvshmem_dist_bits, &nvshmem_sssp_frontier[current],
			&nvshmem_sssp_frontier_count[current], &pending_counter, &bucket_min,
			&bucket_max, &g.nlocalverts};
		CHECK_NVSHMEM_LAUNCH(nvshmemx_collective_launch((const void*)sssp_bucket_scan_kernel,
			grid, block, scan_args, 0, 0));
		CHECK_CUDA(cudaDeviceSynchronize());
		nvshmem_barrier_all();

		unsigned long long local_pending = 0;
		CHECK_CUDA(cudaMemcpy(&local_pending, pending_counter, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
		local_count_64 = (int64_t)local_pending;
		MPI_Allreduce(&local_count_64, &global_count, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
	}

	float *device_export = NULL;
	CHECK_CUDA(cudaMalloc(&device_export, g.nlocalverts * sizeof(float)));
	void *export_args[] = {&nvshmem_dist_bits, &device_export, &g.nlocalverts};
	CHECK_NVSHMEM_LAUNCH(nvshmemx_collective_launch((const void*)sssp_export_dist_kernel,
		grid, block, export_args, 0, 0));
	CHECK_CUDA(cudaDeviceSynchronize());
	CHECK_CUDA(cudaMemcpy(dist, device_export, g.nlocalverts * sizeof(float), cudaMemcpyDeviceToHost));
	CHECK_CUDA(cudaMemcpy(pred, nvshmem_pred, g.nlocalverts * sizeof(int64_t), cudaMemcpyDeviceToHost));
	CHECK_CUDA(cudaFree(device_export));
}

void clean_shortest(float* dist) {
	int i;
	for(i=0;i<g.nlocalverts;i++) dist[i]=-1.0;
	if (nvshmem_dist_bits != NULL) {
		CHECK_CUDA(cudaMemset(nvshmem_sssp_pred_lock, 0, g.nlocalverts * sizeof(int)));
		void *args[] = {&nvshmem_dist_bits, &g.nlocalverts};
		CHECK_NVSHMEM_LAUNCH(nvshmemx_collective_launch((const void*)sssp_clean_dist_kernel,
			dim3(128), dim3(256), args, 0, 0));
		CHECK_CUDA(cudaDeviceSynchronize());
		nvshmem_barrier_all();
	}
}
#endif
