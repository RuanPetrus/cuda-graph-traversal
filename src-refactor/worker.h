#ifndef WORKER_H
#define WORKER_H
#include "base.h"

typedef struct State State;
struct Worker_State {
	Arena cpu_arena;
	Arena gpu_arena;
	i32 rank;
	i32 rank_size;
};

Worker_State worker_init(i32 argc, char **argv, u64 gpu_memory_size);
void worker_finalize(Worker_State *ws);
void worker_abort(Worker_State *ws, i32 return_code);
void *_worker_arena_push_array(Worker_State *ws, u64 local_count, u64 elem_size, u64 alignment, u64 *max_count_out);

#define worker_arena_push_array(ws, count, type, max_count_out) \
	((type *)_worker_arena_push_array((ws), (count), sizeof(type), alignof(type), (max_count_out)))

#endif // WORKER_H
