#ifndef __STDC_CONSTANT_MACROS
#define __STDC_CONSTANT_MACROS
#endif
#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS
#endif

#include "common.h"

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct mrg_state {
	uint_fast32_t z1, z2, z3, z4, z5;
} mrg_state;

typedef struct mrg_transition_matrix {
	uint_fast32_t s, t, u, v, w;
	uint_fast32_t a, b, c, d;
} mrg_transition_matrix;

static mrg_transition_matrix mrg_skip_matrices[24][256];
static int mrg_skip_ready;

static uint_fast32_t mod_add(uint_fast32_t a, uint_fast32_t b) {
	uint_fast32_t x = a + b;
	return (x >= 0x7FFFFFFF) ? (x - 0x7FFFFFFF) : x;
}

static uint_fast32_t mod_mul(uint_fast32_t a, uint_fast32_t b) {
	uint_fast64_t temp = (uint_fast64_t)a * b;
	uint_fast32_t temp2 = (uint_fast32_t)(temp & 0x7FFFFFFF) + (uint_fast32_t)(temp >> 31);
	return (temp2 >= 0x7FFFFFFF) ? (temp2 - 0x7FFFFFFF) : temp2;
}

static uint_fast32_t mod_mac(uint_fast32_t sum, uint_fast32_t a, uint_fast32_t b) {
	uint_fast64_t temp = (uint_fast64_t)a * b + sum;
	uint_fast32_t temp2 = (uint_fast32_t)(temp & 0x7FFFFFFF) + (uint_fast32_t)(temp >> 31);
	return (temp2 >= 0x7FFFFFFF) ? (temp2 - 0x7FFFFFFF) : temp2;
}

static uint_fast32_t mod_mac2(uint_fast32_t sum, uint_fast32_t a, uint_fast32_t b,
		uint_fast32_t c, uint_fast32_t d) {
	return mod_mac(mod_mac(sum, a, b), c, d);
}

static uint_fast32_t mod_mac3(uint_fast32_t sum, uint_fast32_t a, uint_fast32_t b,
		uint_fast32_t c, uint_fast32_t d, uint_fast32_t e, uint_fast32_t f) {
	return mod_mac2(mod_mac(sum, a, b), c, d, e, f);
}

static uint_fast32_t mod_mac4(uint_fast32_t sum, uint_fast32_t a, uint_fast32_t b,
		uint_fast32_t c, uint_fast32_t d, uint_fast32_t e, uint_fast32_t f,
		uint_fast32_t g, uint_fast32_t h) {
	return mod_mac2(mod_mac2(sum, a, b, c, d), e, f, g, h);
}

static uint_fast32_t mod_mul_x(uint_fast32_t a) {
	int_fast32_t result = (int_fast32_t)(a) / 20;
	result = 107374182 * ((int_fast32_t)(a) - result * 20) - result * 7;
	result += (result < 0 ? 0x7FFFFFFF : 0);
	return (uint_fast32_t)result;
}

static uint_fast32_t mod_mul_y(uint_fast32_t a) {
	int_fast32_t result = (int_fast32_t)(a) / 20554;
	result = 104480 * ((int_fast32_t)(a) - result * 20554) - result * 1727;
	result += (result < 0 ? 0x7FFFFFFF : 0);
	return (uint_fast32_t)result;
}

static uint_fast32_t mod_mac_y(uint_fast32_t sum, uint_fast32_t a) {
	return mod_add(sum, mod_mul_y(a));
}

static void mrg_update_cache(mrg_transition_matrix* p) {
	p->a = mod_add(mod_mul_x(p->s), p->t);
	p->b = mod_add(mod_mul_x(p->a), p->u);
	p->c = mod_add(mod_mul_x(p->b), p->v);
	p->d = mod_add(mod_mul_x(p->c), p->w);
}

static void mrg_make_identity(mrg_transition_matrix* result) {
	result->s = result->t = result->u = result->v = 0;
	result->w = 1;
	mrg_update_cache(result);
}

static void mrg_make_A(mrg_transition_matrix* result) {
	result->s = result->t = result->u = result->w = 0;
	result->v = 1;
	mrg_update_cache(result);
}

static void mrg_multiply(const mrg_transition_matrix* m, const mrg_transition_matrix* n,
		mrg_transition_matrix* result) {
	uint_fast32_t rs = mod_mac(mod_mac(mod_mac(mod_mac(mod_mul(m->s, n->d), m->t, n->c), m->u, n->b), m->v, n->a), m->w, n->s);
	uint_fast32_t rt = mod_mac(mod_mac(mod_mac(mod_mac(mod_mul_y(mod_mul(m->s, n->s)), m->t, n->w), m->u, n->v), m->v, n->u), m->w, n->t);
	uint_fast32_t ru = mod_mac(mod_mac(mod_mac(mod_mul_y(mod_mac(mod_mul(m->s, n->a), m->t, n->s)), m->u, n->w), m->v, n->v), m->w, n->u);
	uint_fast32_t rv = mod_mac(mod_mac(mod_mul_y(mod_mac(mod_mac(mod_mul(m->s, n->b), m->t, n->a), m->u, n->s)), m->v, n->w), m->w, n->v);
	uint_fast32_t rw = mod_mac(mod_mul_y(mod_mac(mod_mac(mod_mac(mod_mul(m->s, n->c), m->t, n->b), m->u, n->a), m->v, n->s)), m->w, n->w);
	result->s = rs;
	result->t = rt;
	result->u = ru;
	result->v = rv;
	result->w = rw;
	mrg_update_cache(result);
}

static void mrg_power(const mrg_transition_matrix* m, unsigned int exponent,
		mrg_transition_matrix* result) {
	mrg_transition_matrix current_power_of_2 = *m;
	mrg_make_identity(result);
	while (exponent > 0) {
		if (exponent % 2 == 1) mrg_multiply(result, &current_power_of_2, result);
		mrg_multiply(&current_power_of_2, &current_power_of_2, &current_power_of_2);
		exponent /= 2;
	}
}

static void mrg_init_skip_matrices(void) {
	if (mrg_skip_ready) return;
	mrg_transition_matrix transitions[24];
	for (int byte = 0; byte < 24; ++byte) {
		mrg_make_identity(&mrg_skip_matrices[byte][0]);
		if (byte == 0) mrg_make_A(&transitions[byte]);
		else mrg_power(&transitions[byte - 1], 256, &transitions[byte]);
		for (int value = 1; value < 256; ++value) {
			mrg_power(&transitions[byte], value, &mrg_skip_matrices[byte][value]);
		}
	}
	mrg_skip_ready = 1;
}

static void mrg_apply_transition(const mrg_transition_matrix* mat, const mrg_state* st, mrg_state* r) {
	uint_fast32_t o1 = mod_mac_y(mod_mul(mat->d, st->z1), mod_mac4(0, mat->s, st->z2, mat->a, st->z3, mat->b, st->z4, mat->c, st->z5));
	uint_fast32_t o2 = mod_mac_y(mod_mac2(0, mat->c, st->z1, mat->w, st->z2), mod_mac3(0, mat->s, st->z3, mat->a, st->z4, mat->b, st->z5));
	uint_fast32_t o3 = mod_mac_y(mod_mac3(0, mat->b, st->z1, mat->v, st->z2, mat->w, st->z3), mod_mac2(0, mat->s, st->z4, mat->a, st->z5));
	uint_fast32_t o4 = mod_mac_y(mod_mac4(0, mat->a, st->z1, mat->u, st->z2, mat->v, st->z3, mat->w, st->z4), mod_mul(mat->s, st->z5));
	uint_fast32_t o5 = mod_mac2(mod_mac3(0, mat->s, st->z1, mat->t, st->z2, mat->u, st->z3), mat->v, st->z4, mat->w, st->z5);
	r->z1 = o1;
	r->z2 = o2;
	r->z3 = o3;
	r->z4 = o4;
	r->z5 = o5;
}

static void mrg_orig_step(mrg_state* state) {
	uint_fast32_t new_elt = mod_mac_y(mod_mul_x(state->z1), state->z5);
	state->z5 = state->z4;
	state->z4 = state->z3;
	state->z3 = state->z2;
	state->z2 = state->z1;
	state->z1 = new_elt;
}

static void mrg_skip(mrg_state* state, uint_least64_t exponent_high,
		uint_least64_t exponent_middle, uint_least64_t exponent_low) {
	mrg_init_skip_matrices();
	int byte_index;
	for (byte_index = 0; exponent_low; ++byte_index, exponent_low >>= 8) {
		uint_least8_t val = (uint_least8_t)(exponent_low & 0xFF);
		if (val != 0) mrg_apply_transition(&mrg_skip_matrices[byte_index][val], state, state);
	}
	for (byte_index = 8; exponent_middle; ++byte_index, exponent_middle >>= 8) {
		uint_least8_t val = (uint_least8_t)(exponent_middle & 0xFF);
		if (val != 0) mrg_apply_transition(&mrg_skip_matrices[byte_index][val], state, state);
	}
	for (byte_index = 16; exponent_high; ++byte_index, exponent_high >>= 8) {
		uint_least8_t val = (uint_least8_t)(exponent_high & 0xFF);
		if (val != 0) mrg_apply_transition(&mrg_skip_matrices[byte_index][val], state, state);
	}
}

static void mrg_seed(mrg_state* st, const uint_fast32_t seed[5]) {
	assert(seed[0] < 0x7FFFFFFF && seed[1] < 0x7FFFFFFF && seed[2] < 0x7FFFFFFF && seed[3] < 0x7FFFFFFF && seed[4] < 0x7FFFFFFF);
	assert(seed[0] != 0 || seed[1] != 0 || seed[2] != 0 || seed[3] != 0 || seed[4] != 0);
	st->z1 = seed[0];
	st->z2 = seed[1];
	st->z3 = seed[2];
	st->z4 = seed[3];
	st->z5 = seed[4];
}

static uint_fast32_t mrg_get_uint_orig(mrg_state* state) {
	mrg_orig_step(state);
	return state->z1;
}

static double mrg_get_double_orig(mrg_state* state) {
	return (double)mrg_get_uint_orig(state) * .0000000004656612875245796924106;
}

static float mrg_get_float_orig(mrg_state* state) {
	return (float)mrg_get_uint_orig(state) * .000000000465661287524579692f;
}

extern "C" void make_mrg_seed(uint64_t userseed1, uint64_t userseed2, uint_fast32_t* seed) {
	seed[0] = (uint32_t)(userseed1 & UINT32_C(0x3FFFFFFF)) + 1;
	seed[1] = (uint32_t)((userseed1 >> 30) & UINT32_C(0x3FFFFFFF)) + 1;
	seed[2] = (uint32_t)(userseed2 & UINT32_C(0x3FFFFFFF)) + 1;
	seed[3] = (uint32_t)((userseed2 >> 30) & UINT32_C(0x3FFFFFFF)) + 1;
	seed[4] = (uint32_t)((userseed2 >> 60) << 4) + (uint32_t)(userseed1 >> 60) + 1;
}

extern "C" void make_random_numbers(int64_t nvalues, uint64_t userseed1, uint64_t userseed2,
		int64_t position, double* result) {
	uint_fast32_t seed[5];
	make_mrg_seed(userseed1, userseed2, seed);
	mrg_state st;
	mrg_seed(&st, seed);
	mrg_skip(&st, 2, 0, 2 * (uint64_t)position);
	for (int64_t i = 0; i < nvalues; ++i) result[i] = mrg_get_double_orig(&st);
}

#define INITIATOR_A_NUMERATOR 5700
#define INITIATOR_BC_NUMERATOR 1900
#define INITIATOR_DENOMINATOR 10000
#define SPK_NOISE_LEVEL 0

static int generate_4way_bernoulli(mrg_state* st, int level, int nlevels) {
	(void)level;
	(void)nlevels;
	static const uint32_t limit = (UINT32_C(0x7FFFFFFF) % INITIATOR_DENOMINATOR);
	uint32_t val = mrg_get_uint_orig(st);
	if (val < limit) {
		do {
			val = mrg_get_uint_orig(st);
		} while (val < limit);
	}
	val %= INITIATOR_DENOMINATOR;
	if (val < INITIATOR_BC_NUMERATOR) return 1;
	val = (uint32_t)(val - INITIATOR_BC_NUMERATOR);
	if (val < INITIATOR_BC_NUMERATOR) return 2;
	val = (uint32_t)(val - INITIATOR_BC_NUMERATOR);
	if (val < INITIATOR_A_NUMERATOR) return 0;
	return 3;
}

static uint64_t bitreverse(uint64_t x) {
	x = __builtin_bswap64(x);
	x = ((x >> 4) & UINT64_C(0x0F0F0F0F0F0F0F0F)) | ((x & UINT64_C(0x0F0F0F0F0F0F0F0F)) << 4);
	x = ((x >> 2) & UINT64_C(0x3333333333333333)) | ((x & UINT64_C(0x3333333333333333)) << 2);
	x = ((x >> 1) & UINT64_C(0x5555555555555555)) | ((x & UINT64_C(0x5555555555555555)) << 1);
	return x;
}

static int64_t scramble(int64_t v0, int lgN, uint64_t val0, uint64_t val1) {
	uint64_t v = (uint64_t)v0;
	v += val0 + val1;
	v *= (val0 | UINT64_C(0x4519840211493211));
	v = (bitreverse(v) >> (64 - lgN));
	assert((v >> lgN) == 0);
	v *= (val1 | UINT64_C(0x3050852102C843A5));
	v = (bitreverse(v) >> (64 - lgN));
	assert((v >> lgN) == 0);
	return (int64_t)v;
}

static void make_one_edge(int64_t nverts, int level, int lgN, mrg_state* st,
		packed_edge* result, uint64_t val0, uint64_t val1) {
	int64_t base_src = 0, base_tgt = 0;
	while (nverts > 1) {
		int square = generate_4way_bernoulli(st, level, lgN);
		int src_offset = square / 2;
		int tgt_offset = square % 2;
		assert(base_src <= base_tgt);
		if (base_src == base_tgt && src_offset > tgt_offset) {
			int temp = src_offset;
			src_offset = tgt_offset;
			tgt_offset = temp;
		}
		nverts /= 2;
		++level;
		base_src += nverts * src_offset;
		base_tgt += nverts * tgt_offset;
	}
	write_edge(result, scramble(base_src, lgN, val0, val1), scramble(base_tgt, lgN, val0, val1));
}

extern "C" void generate_kronecker_range(const uint_fast32_t seed[5], int logN,
		int64_t start_edge, int64_t end_edge, packed_edge* edges
#ifdef SSSP
		,float* weights
#endif
		) {
	mrg_state state;
	mrg_seed(&state, seed);
	uint64_t val0, val1;
	{
		mrg_state new_state = state;
		mrg_skip(&new_state, 50, 7, 0);
		val0 = mrg_get_uint_orig(&new_state);
		val0 *= UINT64_C(0xFFFFFFFF);
		val0 += mrg_get_uint_orig(&new_state);
		val1 = mrg_get_uint_orig(&new_state);
		val1 *= UINT64_C(0xFFFFFFFF);
		val1 += mrg_get_uint_orig(&new_state);
	}

	int64_t nverts = (int64_t)1 << logN;
	for (int64_t ei = start_edge; ei < end_edge; ++ei) {
		mrg_state new_state = state;
		mrg_skip(&new_state, 0, (uint64_t)ei, 0);
		make_one_edge(nverts, 0, logN, &new_state, edges + (ei - start_edge), val0, val1);
#ifdef SSSP
		weights[ei - start_edge] = mrg_get_float_orig(&new_state);
#endif
	}
}
