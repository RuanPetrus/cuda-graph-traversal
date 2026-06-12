#include <mpi.h>

#include "graph_generation.h"
#include "mrg_transitions.cu"

#define TUPLE_GRAPH_DUMP_EDGES "graph-refactor.bin"
#define TUPLE_GRAPH_DUMP_WEIGHTS "graph-refactor.bin.weights"

__host__ __device__ static u32 cuda_mod_add(u32 a, u32 b) {
  u32 x = a + b;
  return (x >= 0x7FFFFFFF) ? (x - 0x7FFFFFFF) : x;
}

__host__ __device__ static u32 cuda_mod_mul(u32 a, u32 b) {
  u64 temp = (u64)a * b;
  u32 temp2 = (u32)(temp & 0x7FFFFFFF) + (u32)(temp >> 31);
  return (temp2 >= 0x7FFFFFFF) ? (temp2 - 0x7FFFFFFF) : temp2;
}

__host__ __device__ static u32 cuda_mod_mac(u32 sum, u32 a, u32 b) {
  u64 temp = (u64)a * b + sum;
  u32 temp2 = (u32)(temp & 0x7FFFFFFF) + (u32)(temp >> 31);
  return (temp2 >= 0x7FFFFFFF) ? (temp2 - 0x7FFFFFFF) : temp2;
}

__host__ __device__ static u32 cuda_mod_mac2(u32 sum, u32 a, u32 b, u32 c, u32 d) {
  return cuda_mod_mac(cuda_mod_mac(sum, a, b), c, d);
}

__host__ __device__ static u32 cuda_mod_mac3(u32 sum, u32 a, u32 b, u32 c, u32 d, u32 e, u32 f) {
  return cuda_mod_mac2(cuda_mod_mac(sum, a, b), c, d, e, f);
}

__host__ __device__ static u32 cuda_mod_mac4(u32 sum, u32 a, u32 b, u32 c, u32 d, u32 e, u32 f, u32 g, u32 h) {
  return cuda_mod_mac2(cuda_mod_mac2(sum, a, b, c, d), e, f, g, h);
}

__host__ __device__ static u32 cuda_mod_mul_x(u32 a) {
  i32 result = (i32)(a) / 20;
  result = 107374182 * ((i32)(a) - result * 20) - result * 7;
  result += (result < 0 ? 0x7FFFFFFF : 0);
  return (u32)result;
}

__host__ __device__ static u32 cuda_mod_mul_y(u32 a) {
  i32 result = (i32)(a) / 20554;
  result = 104480 * ((i32)(a) - result * 20554) - result * 1727;
  result += (result < 0 ? 0x7FFFFFFF : 0);
  return (u32)result;
}

__host__ __device__ static u32 cuda_mod_mac_y(u32 sum, u32 a) {
  return cuda_mod_add(sum, cuda_mod_mul_y(a));
}

__host__ __device__ static void cuda_mrg_apply_transition(const Mrg_Transition_Matrix* mat, const Mrg_State* st, Mrg_State* r) {
  u32 o1 = cuda_mod_mac_y(cuda_mod_mul(mat->d, st->z1), cuda_mod_mac4(0, mat->s, st->z2, mat->a, st->z3, mat->b, st->z4, mat->c, st->z5));
  u32 o2 = cuda_mod_mac_y(cuda_mod_mac2(0, mat->c, st->z1, mat->w, st->z2), cuda_mod_mac3(0, mat->s, st->z3, mat->a, st->z4, mat->b, st->z5));
  u32 o3 = cuda_mod_mac_y(cuda_mod_mac3(0, mat->b, st->z1, mat->v, st->z2, mat->w, st->z3), cuda_mod_mac2(0, mat->s, st->z4, mat->a, st->z5));
  u32 o4 = cuda_mod_mac_y(cuda_mod_mac4(0, mat->a, st->z1, mat->u, st->z2, mat->v, st->z3, mat->w, st->z4), cuda_mod_mul(mat->s, st->z5));
  u32 o5 = cuda_mod_mac2(cuda_mod_mac3(0, mat->s, st->z1, mat->t, st->z2, mat->u, st->z3), mat->v, st->z4, mat->w, st->z5);
  r->z1 = o1;
  r->z2 = o2;
  r->z3 = o3;
  r->z4 = o4;
  r->z5 = o5;
}

__host__ __device__ static void cuda_mrg_step(const Mrg_Transition_Matrix* mat, Mrg_State* state) {
  cuda_mrg_apply_transition(mat, state, state);
}

__host__ __device__ static void cuda_mrg_orig_step(Mrg_State* state) {
	u32 new_elt = cuda_mod_mac_y(cuda_mod_mul_x(state->z1), state->z5);
	state->z5 = state->z4;
	state->z4 = state->z3;
	state->z3 = state->z2;
	state->z2 = state->z1;
	state->z1 = new_elt;
}

__host__ __device__ static u32 cuda_mrg_get_uint_orig(Mrg_State* state) {
	cuda_mrg_orig_step(state);
	return state->z1;
}


__host__ __device__ static f32 cuda_mrg_get_float_orig(Mrg_State* state) {
  return (f32)cuda_mrg_get_uint_orig(state) * .000000000465661287524579692f;
}

/* Initiator settings: for faster random number generation, the initiator
 * probabilities are defined as fractions (a = INITIATOR_A_NUMERATOR /
 * INITIATOR_DENOMINATOR, b = c = INITIATOR_BC_NUMERATOR /
 * INITIATOR_DENOMINATOR, d = 1 - a - b - c. */
#define INITIATOR_A_NUMERATOR 5700
#define INITIATOR_BC_NUMERATOR 1900
#define INITIATOR_DENOMINATOR 10000

/* If this macro is defined to a non-zero value, use SPK_NOISE_LEVEL /
 * INITIATOR_DENOMINATOR as the noise parameter to use in introducing noise
 * into the graph parameters.  The approach used is from "A Hitchhiker's Guide
 * to Choosing Parameters of Stochastic Kronecker Graphs" by C. Seshadhri, Ali
 * Pinar, and Tamara G. Kolda (http://arxiv.org/abs/1102.5046v1), except that
 * the adjustment here is chosen based on the current level being processed
 * rather than being chosen randomly. */
#define SPK_NOISE_LEVEL 0
/* #define SPK_NOISE_LEVEL 1000 -- in INITIATOR_DENOMINATOR units */

__device__ static i32 cuda_generate_4way_bernoulli(Mrg_State* st, i32 level, i32 nlevels) {
#if SPK_NOISE_LEVEL == 0
  (void)level;
  (void)nlevels;
#endif
  static const u32 limit = (UINT32_C(0x7FFFFFFF) % INITIATOR_DENOMINATOR);
  u32 val = cuda_mrg_get_uint_orig(st);
  if (val < limit) {
    do {
      val = cuda_mrg_get_uint_orig(st);
    } while (val < limit);
  }
#if SPK_NOISE_LEVEL == 0
  i32 spk_noise_factor = 0;
#else
  i32 spk_noise_factor = 2 * SPK_NOISE_LEVEL * level / nlevels - SPK_NOISE_LEVEL;
#endif
  u32 adjusted_bc_numerator = (u32)(INITIATOR_BC_NUMERATOR + spk_noise_factor);
  val %= INITIATOR_DENOMINATOR;
  if (val < adjusted_bc_numerator) return 1;
  val = (u32)(val - adjusted_bc_numerator);
  if (val < adjusted_bc_numerator) return 2;
  val = (u32)(val - adjusted_bc_numerator);
#if SPK_NOISE_LEVEL == 0
  if (val < INITIATOR_A_NUMERATOR) return 0;
#else
  if (val < INITIATOR_A_NUMERATOR * (INITIATOR_DENOMINATOR - 2 * INITIATOR_BC_NUMERATOR) / (INITIATOR_DENOMINATOR - 2 * adjusted_bc_numerator)) return 0;
#endif
  return 3;
}

__device__ static u64 cuda_bitreverse(u64 x) {
#ifdef FAST_64BIT_ARITHMETIC
  x = ((x & UINT64_C(0x00000000FFFFFFFF)) << 32) | ((x >> 32) & UINT64_C(0x00000000FFFFFFFF));
  x = ((x & UINT64_C(0x0000FFFF0000FFFF)) << 16) | ((x >> 16) & UINT64_C(0x0000FFFF0000FFFF));
  x = ((x & UINT64_C(0x00FF00FF00FF00FF)) << 8) | ((x >> 8) & UINT64_C(0x00FF00FF00FF00FF));
  x = ((x & UINT64_C(0x0F0F0F0F0F0F0F0F)) << 4) | ((x >> 4) & UINT64_C(0x0F0F0F0F0F0F0F0F));
  x = ((x & UINT64_C(0x3333333333333333)) << 2) | ((x >> 2) & UINT64_C(0x3333333333333333));
  x = ((x & UINT64_C(0x5555555555555555)) << 1) | ((x >> 1) & UINT64_C(0x5555555555555555));
  return x;
#else
  u32 h = (u32)(x >> 32);
  u32 l = (u32)(x & UINT32_MAX);
  h = (h >> 16) | (h << 16);
  l = (l >> 16) | (l << 16);
  h = ((h >> 8) & UINT32_C(0x00FF00FF)) | ((h & UINT32_C(0x00FF00FF)) << 8);
  l = ((l >> 8) & UINT32_C(0x00FF00FF)) | ((l & UINT32_C(0x00FF00FF)) << 8);
  h = ((h >> 4) & UINT32_C(0x0F0F0F0F)) | ((h & UINT32_C(0x0F0F0F0F)) << 4);
  l = ((l >> 4) & UINT32_C(0x0F0F0F0F)) | ((l & UINT32_C(0x0F0F0F0F)) << 4);
  h = ((h >> 2) & UINT32_C(0x33333333)) | ((h & UINT32_C(0x33333333)) << 2);
  l = ((l >> 2) & UINT32_C(0x33333333)) | ((l & UINT32_C(0x33333333)) << 2);
  h = ((h >> 1) & UINT32_C(0x55555555)) | ((h & UINT32_C(0x55555555)) << 1);
  l = ((l >> 1) & UINT32_C(0x55555555)) | ((l & UINT32_C(0x55555555)) << 1);
  return ((u64)l << 32) | h;
#endif
}

__device__ static i64 cuda_scramble(i64 v0, int lgN, u64 val0, u64 val1) {
  u64 v = (u64)v0;
  v += val0 + val1;
  v *= (val0 | UINT64_C(0x4519840211493211));
  v = (cuda_bitreverse(v) >> (64 - lgN));
  v *= (val1 | UINT64_C(0x3050852102C843A5));
  v = (cuda_bitreverse(v) >> (64 - lgN));
  return (i64)v;
}

__device__ static void cuda_write_edge(Packed_Edge* p, i64 v0, i64 v1) {
  p->v0 = v0;
  p->v1 = v1;
}

__device__ static void cuda_make_one_edge(i64 nverts, i32 level, i32 lgN, Mrg_State* st, Packed_Edge* result, u64 val0, u64 val1) {
  i64 base_src = 0, base_tgt = 0;
  while (nverts > 1) {
    int square = cuda_generate_4way_bernoulli(st, level, lgN);
    int src_offset = square / 2;
    int tgt_offset = square % 2;
    if (base_src == base_tgt) {
      if (src_offset > tgt_offset) {
        int temp = src_offset;
        src_offset = tgt_offset;
        tgt_offset = temp;
      }
    }
    nverts /= 2;
    ++level;
    base_src += nverts * src_offset;
    base_tgt += nverts * tgt_offset;
  }
  cuda_write_edge(result,
                  cuda_scramble(base_src, lgN, val0, val1),
                  cuda_scramble(base_tgt, lgN, val0, val1));
}

__host__ __device__ static void cuda_mrg_skip(Mrg_State* state, u64 exponent_high, u64 exponent_middle, u64 exponent_low) {
  i32 byte_index;
  for (byte_index = 0; exponent_low; ++byte_index, exponent_low >>= 8) {
    u8 val = (u8)(exponent_low & 0xFF);
    if (val != 0) cuda_mrg_step(&cuda_mrg_skip_matrices[byte_index][val], state);
  }
  for (byte_index = 8; exponent_middle; ++byte_index, exponent_middle >>= 8) {
    u8 val = (u8)(exponent_middle & 0xFF);
    if (val != 0) cuda_mrg_step(&cuda_mrg_skip_matrices[byte_index][val], state);
  }
  for (byte_index = 16; exponent_high; ++byte_index, exponent_high >>= 8) {
    u8 val = (u8)(exponent_high & 0xFF);
    if (val != 0) cuda_mrg_step(&cuda_mrg_skip_matrices[byte_index][val], state);
  }
}

__global__ static void generate_kronecker_kernel(Mrg_State base_state, i32 logN,
    i64 start_edge, i64 edge_count, Packed_Edge* edges, u64 val0, u64 val1, f32* weights) {
  i64 tid    = (i64)blockIdx.x * blockDim.x + threadIdx.x;
  i64 stride = (i64)blockDim.x * gridDim.x;
  i64 nverts = (i64)1 << logN;

  for (i64 i = tid; i < edge_count; i += stride) {
    i64 ei = start_edge + i;
    Mrg_State new_state = base_state;
    cuda_mrg_skip(&new_state, 0, (u64)ei, 0);
    cuda_make_one_edge(nverts, 0, logN, &new_state, edges + i, val0, val1);
    weights[i] = cuda_mrg_get_float_orig(&new_state);
  }
}

void mrg_seed(Mrg_State* st, u32 seed[5]) {
  st->z1 = seed[0];
  st->z2 = seed[1];
  st->z3 = seed[2];
  st->z4 = seed[3];
  st->z5 = seed[4];
}

void make_mrg_seed(u64 userseed1, u64 userseed2, u32* seed) {
	seed[0] = (u32)(userseed1 & UINT32_C(0x3FFFFFFF)) + 1;
	seed[1] = (u32)((userseed1 >> 30) & UINT32_C(0x3FFFFFFF)) + 1;
	seed[2] = (u32)(userseed2 & UINT32_C(0x3FFFFFFF)) + 1;
	seed[3] = (u32)((userseed2 >> 30) & UINT32_C(0x3FFFFFFF)) + 1;
	seed[4] = (u32)((userseed2 >> 60) << 4) + (u32)(userseed1 >> 60) + 1;
}

// TODO(ruan): Do some refactor here
void generate_kronecker_range(Arena *gpu_arena, 
							  u32 rank,
		                      u32 seed[5] /* All values in [0, 2^31 - 1), not all zero */,
                              u32 logN /* In base 2 */,
                              i64 start_edge, i64 end_edge,
                              Tuple_Graph* tg) {
  Mrg_State state;
  i64 edge_count = end_edge - start_edge;
  tg->edges_size = (u64)edge_count;
  tg->max_edges_size = (u64)edge_count;
  if (edge_count <= 0) return;

  i32 device_count = 0;
  CHECK_CUDA(cudaGetDeviceCount(&device_count));
  if (device_count <= 0) {
	ERROR("Rank x: no CUDA devices available for graph generation\n");
    ABORT();
  }
  CHECK_CUDA(cudaSetDevice(rank % device_count));
  mrg_seed(&state, seed);

  u64 val0, val1; /* Values for scrambling */
  {
    Mrg_State new_state = state;
    cuda_mrg_skip(&new_state, 50, 7, 0);
    val0 = cuda_mrg_get_uint_orig(&new_state);
    val0 *= UINT64_C(0xFFFFFFFF);
    val0 += cuda_mrg_get_uint_orig(&new_state);
    val1 = cuda_mrg_get_uint_orig(&new_state);
    val1 *= UINT64_C(0xFFFFFFFF);
    val1 += cuda_mrg_get_uint_orig(&new_state);
  }

  tg->edges   = arena_push_array(gpu_arena, edge_count, Packed_Edge);
  tg->weights = arena_push_array(gpu_arena, edge_count, f32);

  i32 threads_per_block = 256;
  i64 blocks_needed = (edge_count + threads_per_block - 1) / threads_per_block;
  i32 blocks = (blocks_needed > 65535) ? 65535 : (i32)blocks_needed;

  generate_kronecker_kernel<<<blocks, threads_per_block>>>(state, logN, start_edge,
      edge_count, tg->edges, val0, val1, tg->weights);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
}

// This part was generated by a clunker, i do not know if its correct

void tuple_graph_dump(const Tuple_Graph* tg, i64 start_edge) {
  size_t edge_bytes = (size_t)tg->edges_size * sizeof(Packed_Edge);
  size_t weight_bytes = (size_t)tg->edges_size * sizeof(f32);
  Packed_Edge* host_edges = edge_bytes ? (Packed_Edge*)malloc(edge_bytes) : NULL;
  f32* host_weights = weight_bytes ? (f32*)malloc(weight_bytes) : NULL;
  if ((edge_bytes && !host_edges) || (weight_bytes && !host_weights)) {
    ERROR("Failed to allocate host graph dump buffers");
    ABORT();
  }

  if (edge_bytes) CHECK_CUDA(cudaMemcpy(host_edges, tg->edges, edge_bytes, cudaMemcpyDeviceToHost));
  if (weight_bytes) CHECK_CUDA(cudaMemcpy(host_weights, tg->weights, weight_bytes, cudaMemcpyDeviceToHost));

  MPI_File edge_file;
  MPI_File weight_file;
  MPI_File_open(MPI_COMM_WORLD, TUPLE_GRAPH_DUMP_EDGES,
                MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &edge_file);
  MPI_File_open(MPI_COMM_WORLD, TUPLE_GRAPH_DUMP_WEIGHTS,
                MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &weight_file);

  MPI_File_set_size(edge_file, (MPI_Offset)tg->nglobaledges * (MPI_Offset)sizeof(Packed_Edge));
  MPI_File_set_size(weight_file, (MPI_Offset)tg->nglobaledges * (MPI_Offset)sizeof(f32));

  if (edge_bytes) {
    MPI_File_write_at(edge_file, (MPI_Offset)start_edge * (MPI_Offset)sizeof(Packed_Edge),
                      host_edges, (int)edge_bytes, MPI_BYTE, MPI_STATUS_IGNORE);
  }
  if (weight_bytes) {
    MPI_File_write_at(weight_file, (MPI_Offset)start_edge * (MPI_Offset)sizeof(f32),
                      host_weights, (int)weight_bytes, MPI_BYTE, MPI_STATUS_IGNORE);
  }

  MPI_File_close(&edge_file);
  MPI_File_close(&weight_file);
  free(host_edges);
  free(host_weights);
}

void tuple_graph_load(Arena* gpu_arena, Tuple_Graph* tg, i64 start_edge, i64 edge_count) {
  tg->edges_size = (u64)edge_count;
  tg->max_edges_size = (u64)edge_count;
  tg->edges = edge_count > 0 ? arena_push_array(gpu_arena, edge_count, Packed_Edge) : NULL;
  tg->weights = edge_count > 0 ? arena_push_array(gpu_arena, edge_count, f32) : NULL;

  size_t edge_bytes = (size_t)edge_count * sizeof(Packed_Edge);
  size_t weight_bytes = (size_t)edge_count * sizeof(f32);
  Packed_Edge* host_edges = edge_bytes ? (Packed_Edge*)malloc(edge_bytes) : NULL;
  f32* host_weights = weight_bytes ? (f32*)malloc(weight_bytes) : NULL;
  if ((edge_bytes && !host_edges) || (weight_bytes && !host_weights)) {
    ERROR("Failed to allocate host graph load buffers");
    ABORT();
  }

  MPI_File edge_file;
  MPI_File weight_file;
  MPI_File_open(MPI_COMM_WORLD, TUPLE_GRAPH_DUMP_EDGES,
                MPI_MODE_RDONLY, MPI_INFO_NULL, &edge_file);
  MPI_File_open(MPI_COMM_WORLD, TUPLE_GRAPH_DUMP_WEIGHTS,
                MPI_MODE_RDONLY, MPI_INFO_NULL, &weight_file);

  if (edge_bytes) {
    MPI_File_read_at(edge_file, (MPI_Offset)start_edge * (MPI_Offset)sizeof(Packed_Edge),
                     host_edges, (int)edge_bytes, MPI_BYTE, MPI_STATUS_IGNORE);
  }
  if (weight_bytes) {
    MPI_File_read_at(weight_file, (MPI_Offset)start_edge * (MPI_Offset)sizeof(f32),
                     host_weights, (int)weight_bytes, MPI_BYTE, MPI_STATUS_IGNORE);
  }

  MPI_File_close(&edge_file);
  MPI_File_close(&weight_file);

  if (edge_bytes) CHECK_CUDA(cudaMemcpy(tg->edges, host_edges, edge_bytes, cudaMemcpyHostToDevice));
  if (weight_bytes) CHECK_CUDA(cudaMemcpy(tg->weights, host_weights, weight_bytes, cudaMemcpyHostToDevice));
  free(host_edges);
  free(host_weights);
}
