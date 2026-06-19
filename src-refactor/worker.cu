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
	i32 device = ws.rank % device_count;
	CHECK_CUDA(cudaSetDevice(device));
	if (gpu_memory_size == 0) {
		MPI_Comm shared_comm;
		MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, ws.rank, MPI_INFO_NULL, &shared_comm);

		MPI_Comm device_comm;
		MPI_Comm_split(shared_comm, device, ws.rank, &device_comm);
		i32 pes_on_device = 1;
		MPI_Comm_size(device_comm, &pes_on_device);

		size_t free_memory = 0;
		size_t total_memory = 0;
		CHECK_CUDA(cudaMemGetInfo(&free_memory, &total_memory));
		u64 local_free_memory = (u64)free_memory / (u64)pes_on_device;
		MPI_Allreduce(&local_free_memory, &gpu_memory_size, 1, MPI_UINT64_T, MPI_MIN, MPI_COMM_WORLD);

		MPI_Comm_free(&device_comm);
		MPI_Comm_free(&shared_comm);

		u64 reserve_memory = MAX(gpu_memory_size / 2, MEGABYTE(512));
		gpu_memory_size = gpu_memory_size > reserve_memory ? gpu_memory_size - reserve_memory : gpu_memory_size / 2;
	}

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

	ws.cpu_arena = arena_create_cpu(MEGABYTE(256));
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

void worker_kernel_launch(Worker_State *ws, const void *kernel, void **args, u64 work_count, i32 threads_per_block, const char *label) {
	dim3 block_dims(threads_per_block);
	int max_blocks = 0;
	int status = nvshmemx_collective_launch_query_gridsize(kernel, block_dims, args, 0, &max_blocks);
	if (status != 0 || max_blocks <= 0) {
		ERROR("NVSHMEM collective launch grid size query failed for %s", label);
		worker_abort(ws, 1);
	}

	u64 blocks_needed = INT_CEIL(work_count, (u64)threads_per_block);
	if (blocks_needed == 0) blocks_needed = 1;
	i32 blocks = blocks_needed > (u64)max_blocks ? max_blocks : (i32)blocks_needed;
	status = nvshmemx_collective_launch(kernel, dim3(blocks), block_dims, args, 0, 0);
	if (status != 0) {
		ERROR("NVSHMEM collective launch failed for %s", label);
		worker_abort(ws, 1);
	}
	CHECK_CUDA(cudaDeviceSynchronize());
}

void *_worker_arena_push_array(Worker_State *ws, u64 local_count, u64 elem_size, u64 alignment, u64 *max_count_out) {
	u64 max_count = 0;
	MPI_Allreduce(&local_count, &max_count, 1, MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD);
	if (max_count_out) *max_count_out = max_count;
	return _arena_push(&ws->gpu_arena, max_count * elem_size, alignment);
}
