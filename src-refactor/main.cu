#include "worker.h"
#include "graph_generation.h"

#define GPU_MEMORY_SIZE GIGABYTE(2)

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

	worker_finalize(&ws);
}
