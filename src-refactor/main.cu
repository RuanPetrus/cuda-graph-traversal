#include <mpi.h>
#include "graph_generation.h"

#define GPU_MEMORY_SIZE GIGABYTE(2)

i32 main(i32 argc, char **argv) {
	MPI_Init(&argc,&argv);
	i32 rank, rank_size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &rank_size);

	i32 SCALE = 16;
	i32 edgefactor = 16;
	if (argc >= 2) SCALE = atoi(argv[1]);
	if (argc >= 3) edgefactor = atoi(argv[2]);
	if (argc <= 1 || argc >= 4 || SCALE == 0 || edgefactor == 0) {
		if (rank == 0) {
			fprintf(stderr, "Usage: %s SCALE edgefactor\n  SCALE = log_2(# vertices) [integer, required]\n  edgefactor = (# edges) / (# vertices) = .5 * (average vertex degree) [integer, defaults to 16]\n(Random number seed and Kronecker initiator are in graph500_runner.c)\n", argv[0]);
		}
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	Arena gpu_arena = arena_create_gpu(GPU_MEMORY_SIZE);
	u64 seed1 = 2, seed2 = 3;

	Tuple_Graph tg = {0};
	tg.nglobaledges = (i64)(edgefactor) << SCALE;

	u32 seed[5];
	make_mrg_seed(seed1, seed2, seed);
	{
		i64 edge_per_rank_count = INT_CEIL(tg.nglobaledges, rank_size);
		i64 start_edge_index = min(edge_per_rank_count * rank, tg.nglobaledges);
		i64 end_edge_index = min(start_edge_index + edge_per_rank_count, tg.nglobaledges);
		generate_kronecker_range(&gpu_arena, rank, seed, SCALE, start_edge_index, end_edge_index, &tg);
	}
	// tuple_graph_dump(&tg, start_edge_index); // Use to check against old implementation

	arena_release_gpu(&gpu_arena);
	MPI_Finalize();
}
