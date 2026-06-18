#ifndef TRAVERSAL_H
#define TRAVERSAL_H

#include "base.h"
#include "worker.h"
#include "graph_generation.h"

typedef struct Oned_Graph Oned_Graph;
struct Oned_Graph {
	u64 nlocalverts;
	u64 nlocaledges;
	u64 nglobalverts, notisolated;
	u64 *rowstarts;
	u64 *column;
	f32 *weights;
};

typedef struct Bfs_State Bfs_State;
struct Bfs_State {
	i64 *pred;
	i64 *dist;
	i32 *frontier[2];
	i32 *frontier_count[2];
	u64 max_nlocalverts;
};

Oned_Graph oned_graph_from_tuple_graph(Worker_State *ws, Tuple_Graph* tg, u64 nglobalverts);
b32 oned_graph_is_vertex_isolated(Worker_State *ws, const Oned_Graph *g, u64 global_vertex);
Bfs_State oned_graph_bfs_create(Worker_State *ws, const Oned_Graph *g);
void oned_graph_bfs_clear(Worker_State *ws, Bfs_State *bfs);
i64 *bfs_compute_dist_from_pred(Worker_State *ws, const Oned_Graph *g, Bfs_State *bfs, u64 root);
void oned_graph_bfs_destroy(Bfs_State *bfs);
void oned_graph_free(Oned_Graph *g);

#endif // TRAVERSAL_H
