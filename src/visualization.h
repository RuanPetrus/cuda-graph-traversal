#ifndef VISUALIZATION_H
#define VISUALIZATION_H

#include "traversal.h"

void tuple_graph_write_dot_files(Worker_State *ws, const Tuple_Graph *tg, const char *output_dir);
void oned_graph_write_dot_files(Worker_State *ws, const Oned_Graph *g, const char *output_dir);
void bfs_write_dot_files(Worker_State *ws, const Oned_Graph *g, const Bfs_State *bfs, u64 root, const char *output_dir);
void bfs_write_dot_files_for_root(Worker_State *ws, const Oned_Graph *g, const Bfs_State *bfs, u64 root, i32 root_idx, const char *output_dir);

#endif // VISUALIZATION_H
