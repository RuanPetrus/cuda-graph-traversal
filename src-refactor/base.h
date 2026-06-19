#ifndef BASE_H
#define BASE_H
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stddef.h>
#include <stdalign.h>
#include <cuda_runtime.h>

typedef int8_t i8;
typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef int32_t b32;
typedef int8_t  b8;

typedef float f32;
typedef double f64;

#define true  1
#define false 0

#define ABS(x) ((x) >= 0 ? (x) : -(x))
#define MAX(x, y) ((x) >= (y) ? (x) : (y))
#define MIN(x, y) ((x) <= (y) ? (x) : (y))

#define GIGABYTE(x) ((x)*1024L*1024L*1024L)
#define MEGABYTE(x) ((x)*1024L*1024L)
#define KILOBYTE(x) ((x)*1024L)

#define INT_CEIL(a, b) (((a) + (b) -1) / (b))

#define ERROR(...)                         \
    do {                                   \
        fprintf(stderr, __VA_ARGS__);      \
        fputc('\n', stderr);               \
    } while (0)

#define ABORT() exit(EXIT_FAILURE) 

#define CHECK_CUDA(call) do { \
  cudaError_t err_ = (call); \
  if (err_ != cudaSuccess) { \
    ERROR("%s:%d: %s failed: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(err_)); \
    ABORT(); \
  } \
} while (0)

u64 mem_align_forward(u64 size, u64 alignment);

#define ARENA_DEFAULT_ALIGNMENT alignof(max_align_t)
#define ARENA_RESERVED_CAPACITY GIGABYTE(10)

typedef struct Arena Arena;
struct Arena {
	u64 reserved;
	u64 used;
	void *base;
};

Arena arena_create_cpu(u64 capacity);
Arena arena_create_gpu(u64 capacity);
Arena arena_from_arena(Arena *arena, u64 capacity);
void arena_release_cpu(Arena *arena);
void arena_release_gpu(Arena *arena);

void *_arena_push(Arena *arena, u64 size, u64 alignment);

#define arena_push(arena, size)              _arena_push(arena, size, ARENA_DEFAULT_ALIGNMENT)
#define arena_push_struct(arena, type)       ((type *)_arena_push(arena, sizeof(type), alignof(type)))
#define arena_push_array(arena, count, type) ((type *)_arena_push(arena, (count) * sizeof(type), alignof(type)))

void arena_clear(Arena *arena);

#endif // BASE_H
