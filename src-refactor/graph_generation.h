#ifndef GRAPH_GENERATION_H
#define GRAPH_GENERATION_H
#include "base.h"

typedef struct Packed_Edge Packed_Edge;
struct Packed_Edge {
	u64 v0;
	u64 v1;
};

typedef struct Tuple_Graph Tuple_Graph;
struct Tuple_Graph {
	Packed_Edge* edges;
	u64 edges_size;
	u64 max_edges_size;
	u64 nglobaledges;
	f32* weights;
};

typedef struct Mrg_Transition_Matrix Mrg_Transition_Matrix;
struct Mrg_Transition_Matrix {
  u32 s, t, u, v, w;
  u32 a, b, c, d;
};

typedef struct Mrg_State Mrg_State;
struct Mrg_State {
  u32 z1, z2, z3, z4, z5;
};

void generate_kronecker_range(Arena *gpu_arena, 
							  u32 rank,
		                      u32 seed[5] /* All values in [0, 2^31 - 1), not all zero */,
                              u32 logN /* In base 2 */,
                              i64 start_edge, i64 end_edge,
                              Tuple_Graph* tg);

void make_mrg_seed(u64 userseed1, u64 userseed2, u32* seed);

void tuple_graph_dump(const Tuple_Graph* tg, i64 start_edge);
void tuple_graph_load(Arena* gpu_arena, Tuple_Graph* tg, i64 start_edge, i64 edge_count);

#endif // GRAPH_GENERATION_H
