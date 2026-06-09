#include "base.h"


typedef struct Oned_Graph Oned_Graph;
struct Oned_Graph {
	u64 nlocalverts;
	u64 max_nlocalverts;
	u64 nlocaledges;
	u32 lg_nglobalverts;
	u64 nglobalverts,notisolated;
	u64 *rowstarts;
	u64 *column;
	f32 *weights;
};

void oned_graph_from_tuple_graph(const Tuple_Graph* tg, Oned_Graph *g);
void oned_graph_free(Oned_Graph *g);


void oned_graph_from_tuple_graph(const Tuple_Graph* tg, Oned_Graph *g) {
	
}
