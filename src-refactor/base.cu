#include "base.h"
#include <string.h>
#include <nvshmem.h>

u64 mem_align_forward(u64 size, u64 alignment) {
	return (size + alignment - 1) & ~(alignment - 1);
}

void *_arena_push(Arena *arena, u64 size, u64 alignment) {
	if (!arena->base) {
		ERROR("Arena was not initialized");
		return 0;
	}
	u64 used_aligned = mem_align_forward(arena->used, alignment);
	u64 new_used = used_aligned + size;

	if (new_used > arena->reserved) {
		ERROR("Arena out of space");
		return 0;
	}
	void *ptr = (u8 *)arena->base + used_aligned;
	arena->used = new_used;
	return ptr;
}

Arena arena_create_cpu(u64 capacity) {
	u64 aligned_capacity = mem_align_forward(capacity, ARENA_DEFAULT_ALIGNMENT);
	return (Arena){
		.reserved = aligned_capacity,
		.used = 0,
		.base = malloc(aligned_capacity),
	};
}

Arena arena_create_gpu(u64 capacity) {
	u64 aligned_capacity = mem_align_forward(capacity, ARENA_DEFAULT_ALIGNMENT);
	Arena arena = {
		.reserved = aligned_capacity,
		.used = 0,
	};
	arena.base = nvshmem_malloc(aligned_capacity);
	if (!arena.base && aligned_capacity != 0) {
		ERROR("Failed to allocate NVSHMEM arena");
		ABORT();
	}
	return arena;
}

void arena_release_cpu(Arena *arena) {
    if (arena->base) {
        free(arena->base);
        arena->reserved = 0;
        arena->used = 0;
        arena->base = 0;
    }
}

void arena_release_gpu(Arena *arena) {
    if (arena->base) {
        nvshmem_free(arena->base);
        arena->reserved = 0;
        arena->used = 0;
        arena->base = 0;
    }
}

void arena_clear(Arena *arena) {
	arena->used = 0;
}
