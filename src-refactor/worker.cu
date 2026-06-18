#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include "worker.h"

Worker_State worker_init(i32 argc, char **argv, u64 gpu_memory_size) {
	MPI_Init(&argc,&argv);
	Worker_State ws = {0};
	MPI_Comm_rank(MPI_COMM_WORLD, &ws.rank);
	MPI_Comm_size(MPI_COMM_WORLD, &ws.rank_size);

	i32 device_count = 0;
	CHECK_CUDA(cudaGetDeviceCount(&device_count));
	if (device_count <= 0) {
		ERROR("Rank %d: no CUDA devices available", ws.rank);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	CHECK_CUDA(cudaSetDevice(ws.rank % device_count));

	if (!getenv("NVSHMEM_SYMMETRIC_SIZE")) {
		char symmetric_size[32];
		snprintf(symmetric_size, sizeof(symmetric_size), "%llu", (unsigned long long)gpu_memory_size);
		setenv("NVSHMEM_SYMMETRIC_SIZE", symmetric_size, 1);
	}

	nvshmemx_init_attr_t attr;
	memset(&attr, 0, sizeof(attr));
	MPI_Comm mpi_comm = MPI_COMM_WORLD;
	attr.mpi_comm = &mpi_comm;
	nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);

	ws.cpu_arena = arena_create_cpu(gpu_memory_size);
	ws.gpu_arena = arena_create_gpu(gpu_memory_size);
	return ws;
}

void worker_finalize(Worker_State *ws) {
	arena_release_gpu(&ws->gpu_arena);
	arena_release_cpu(&ws->cpu_arena);
	nvshmem_finalize();
	MPI_Finalize();
}

void worker_abort(Worker_State *ws, i32 return_code) {
	worker_finalize(ws);
	MPI_Abort(MPI_COMM_WORLD, return_code);
}

void *_worker_arena_push_array(Worker_State *ws, u64 local_count, u64 elem_size, u64 alignment, u64 *max_count_out) {
	u64 max_count = 0;
	MPI_Allreduce(&local_count, &max_count, 1, MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD);
	if (max_count_out) *max_count_out = max_count;
	return _arena_push(&ws->gpu_arena, max_count * elem_size, alignment);
}
