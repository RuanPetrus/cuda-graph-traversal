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

Oned_Graph oned_graph_from_tuple_graph(Worker_State *ws, Tuple_Graph* tg, u64 nglobalverts);
void oned_graph_free(Oned_Graph *g);

#endif // TRAVERSAL_H
