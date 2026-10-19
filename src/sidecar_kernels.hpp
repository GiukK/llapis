// sidecar_kernels.hpp -- the CPU cold executor's own dot-product kernels (src/moe_sidecar.cpp), kept
// in a header so tests/test_sidecar_kernels.cpp can check them BIT FOR BIT against ggml's vec_dot and
// time them without a GPU or a model. Each kernel decodes a weight row ONCE and applies it to N
// activations (the tokens of one MTP verify batch that route to the same expert), with the integer ops
// and the per-(row, activation) accumulation order of the ggml kernel it replaces.
#pragma once

#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <immintrin.h>
#endif

namespace sck {

// fp16 -> fp32 with the F16C instruction, inline (exact, like every fp16 conversion). ggml_fp16_to_fp32
// is an exported function: one DLL call per weight block in the hot loop.
static inline float sck_fp16(uint16_t h) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) h))); }

// ---- decode-once-dot-N for IQ2_XXS x Q8_K -------------------------
// Under MTP one fire carries spec_k tokens (Pmax = spec_k x top_k), so two DIFFERENT tokens
// can route cold to the SAME expert -- measured 24.1% of cold pairs. The grid/sign/scale
// LOOKUP depends only on the weight row, never on the activation, and is ~84% of a row's cost
// (back-solved). Decoding once and applying to N activations was measured bit-exact and
// 42.2%/55.7%/63.9% faster than N x vec_dot at N=2/3/4, at this exact TILE_ROWS granularity,
// on real pages. This is that kernel, ported unchanged.
//
// Bit-exactness is BY CONSTRUCTION, not by tuning: identical integer ops and identical
// per-(row,activation) accumulation order to ggml_vec_dot_iq2_xxs_q8_K -- the decode is merely
// hoisted out of the activation loop. Any edit here must preserve that: a reordering that looks
// harmless changes the logits, and the difference is small enough to read as sampling noise.
#pragma pack(push, 1)
struct blk_iq2_xxs { uint16_t d; uint16_t qs[32]; };   // fp16 d + 32 u16 = 66 B / 256 weights
#pragma pack(pop)
struct blk_q8_K { float d; int8_t qs[256]; int16_t bsums[16]; };
static_assert(sizeof(blk_iq2_xxs) == 66,  "iq2_xxs block layout");
static_assert(sizeof(blk_q8_K)    == 292, "q8_K block layout");

static uint64_t g_grid[256];
static uint64_t g_signs[128];
static bool     g_tables_ok = false;

// AVX2 gate: MSVC emits VEX for explicit intrinsics regardless of /arch, so a non-AVX2 host
// would fault rather than fall back. ggml's own iq2_xxs path is AVX2 on this build, so this
// should never fire in practice -- it exists so the fallback below is real, not theoretical.
static bool cpu_has_avx2() {
#if defined(_MSC_VER)
    int r[4];
    __cpuid(r, 0);
    if (r[0] < 7) return false;
    __cpuidex(r, 7, 0);
    return (r[1] & (1 << 5)) != 0;          // EBX bit 5 = AVX2
#else
    return __builtin_cpu_supports("avx2");
#endif
}

// Rebuild ggml's iq2xxs_grid / keven_signs tables (both are static inside ggml-cpu, not
// exported). Same reconstruction the microbench validated: signs are closed-form; the grid is
// recovered through ggml's OWN dequantizer so it can never drift from the library's table.
static void build_tables() {
    for (uint32_t j = 0; j < 128; ++j) {
#if defined(_MSC_VER)
        const uint32_t pc = __popcnt(j);
#else
        const uint32_t pc = (uint32_t) __builtin_popcount(j);
#endif
        const uint32_t ks = j | ((pc & 1) << 7);
        uint64_t v = 0;
        for (int k = 0; k < 8; ++k)
            v |= (uint64_t) (((ks >> k) & 1) ? (uint8_t) 0xFF : (uint8_t) 0x01) << (8 * k);
        g_signs[j] = v;
    }
    const auto * tr = ggml_get_type_traits(GGML_TYPE_IQ2_XXS);
    if (!tr || !tr->to_float) return;
    for (int base = 0; base < 256; base += 8) {
        blk_iq2_xxs b;
        b.d = 0x3C00;                                    // fp16 1.0
        for (int p = 0; p < 8; ++p) {
            const uint32_t g0 = (uint32_t) (base + p) & 0xFF;
            const uint32_t aux0 = g0 | (g0 << 8) | (g0 << 16) | (g0 << 24);
            memcpy(&b.qs[p * 4], &aux0, 4);
            memset(&b.qs[p * 4 + 2], 0, 4);
        }
        float out[256];
        tr->to_float(&b, out, 256);
        for (int p = 0; p < 8; ++p) {
            uint64_t v = 0;
            for (int k = 0; k < 8; ++k) {
                const int q = (int) lrintf(out[p * 32 + k] * 8.0f);   // to_float applies 0.125f
                v |= (uint64_t) (uint8_t) q << (8 * k);
            }
            g_grid[base + p] = v;
        }
    }
    g_tables_ok = cpu_has_avx2();
}

template <int NACT>
static void iq2xxs_row_decodeN(const char * wr, const blk_q8_K * const * ys, float * outs, int nsb) {
    const blk_iq2_xxs * x = (const blk_iq2_xxs *) wr;
    __m256 accumf[NACT];
    for (int k = 0; k < NACT; ++k) accumf[k] = _mm256_setzero_ps();
    uint32_t aux32[4];
    const uint8_t * aux8 = (const uint8_t *) aux32;
    for (int i = 0; i < nsb; ++i) {
        const uint16_t * q2 = x[i].qs;
        __m256i sumi1[NACT], sumi2[NACT];
        for (int k = 0; k < NACT; ++k) { sumi1[k] = _mm256_setzero_si256(); sumi2[k] = _mm256_setzero_si256(); }
        for (int ib64 = 0; ib64 < 4; ++ib64) {
            memcpy(aux32, q2, 16); q2 += 8;
            // ---- DECODE ONCE: depends on the weight row only ----
            const __m256i q2_1 = _mm256_set_epi64x(g_grid[aux8[ 3]], g_grid[aux8[ 2]], g_grid[aux8[1]], g_grid[aux8[0]]);
            const __m256i q2_2 = _mm256_set_epi64x(g_grid[aux8[11]], g_grid[aux8[10]], g_grid[aux8[9]], g_grid[aux8[8]]);
            const __m256i s2_1 = _mm256_set_epi64x(g_signs[(aux32[1] >> 21) & 127], g_signs[(aux32[1] >> 14) & 127],
                                                   g_signs[(aux32[1] >>  7) & 127], g_signs[(aux32[1] >>  0) & 127]);
            const __m256i s2_2 = _mm256_set_epi64x(g_signs[(aux32[3] >> 21) & 127], g_signs[(aux32[3] >> 14) & 127],
                                                   g_signs[(aux32[3] >>  7) & 127], g_signs[(aux32[3] >>  0) & 127]);
            const __m256i scale1 = _mm256_set1_epi16((short) (2 * (aux32[1] >> 28) + 1));
            const __m256i scale2 = _mm256_set1_epi16((short) (2 * (aux32[3] >> 28) + 1));
            // ---- APPLY N TIMES: same op sequence as the base kernel, per activation ----
            for (int k = 0; k < NACT; ++k) {
                const int8_t * q8b = ys[k][i].qs + ib64 * 64;
                const __m256i q8_1 = _mm256_loadu_si256((const __m256i *) q8b);
                const __m256i q8_2 = _mm256_loadu_si256((const __m256i *) (q8b + 32));
                const __m256i dot1 = _mm256_maddubs_epi16(q2_1, _mm256_sign_epi8(q8_1, s2_1));
                const __m256i dot2 = _mm256_maddubs_epi16(q2_2, _mm256_sign_epi8(q8_2, s2_2));
                sumi1[k] = _mm256_add_epi32(sumi1[k], _mm256_madd_epi16(dot1, scale1));
                sumi2[k] = _mm256_add_epi32(sumi2[k], _mm256_madd_epi16(dot2, scale2));
            }
        }
        const float d = sck_fp16(x[i].d);   // F16C, exact (ggml_fp16_to_fp32 is a DLL call per block)
        for (int k = 0; k < NACT; ++k) {
            const float dk = d * ys[k][i].d;
            accumf[k] = _mm256_fmadd_ps(_mm256_set1_ps(dk),
                        _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1[k], sumi2[k])), accumf[k]);
        }
    }
    for (int k = 0; k < NACT; ++k) {
        __m128 s = _mm_add_ps(_mm256_castps256_ps128(accumf[k]), _mm256_extractf128_ps(accumf[k], 1));
        s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        s = _mm_add_ss(s, _mm_movehdup_ps(s));
        outs[k] = 0.125f * _mm_cvtss_f32(s);
    }
}


// ---- decode-once-dot-N for IQ4_NL x Q8_0 (the down projection, every layer) ---------------------
// Same idea for the 4-bit non-linear down weights: the nibble -> int8 table lookup depends on the
// weight row only. Mirrors ggml_vec_dot_iq4_nl_q8_0's AVX2 path exactly: blocks in pairs, two float
// accumulators per activation, the scale d_y*d_x formed in float, one hsum at the end. Requires an
// even block count (d_in = 704 = 22 blocks here); the caller checks.
#pragma pack(push, 1)
struct blk_iq4_nl { uint16_t d; uint8_t qs[16]; };   // fp16 d + 16 B = 18 B / 32 weights
struct blk_q8_0   { uint16_t d; int8_t  qs[32]; };   // fp16 d + 32 B = 34 B / 32 weights
#pragma pack(pop)
static_assert(sizeof(blk_iq4_nl) == 18, "iq4_nl block layout");
static_assert(sizeof(blk_q8_0)   == 34, "q8_0 block layout");

static const int8_t k_iq4nl_values[16] = { -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 };

static inline float sck_hsum_float_8(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

template <int NACT>
static void iq4nl_row_decodeN(const char * wr, const blk_q8_0 * const * ys, float * outs, int nb) {
    const blk_iq4_nl * x = (const blk_iq4_nl *) wr;
    const __m128i values128 = _mm_loadu_si128((const __m128i *) k_iq4nl_values);
    const __m128i m4b  = _mm_set1_epi8(0x0f);
    const __m256i mone = _mm256_set1_epi16(1);
    __m256 acc1[NACT], acc2[NACT];
    for (int k = 0; k < NACT; ++k) { acc1[k] = _mm256_setzero_ps(); acc2[k] = _mm256_setzero_ps(); }
    for (int ib = 0; ib + 1 < nb; ib += 2) {
        // ---- DECODE ONCE ----
        const __m128i q4bits_1 = _mm_loadu_si128((const __m128i *) x[ib + 0].qs);
        const __m128i q4bits_2 = _mm_loadu_si128((const __m128i *) x[ib + 1].qs);
        const __m256i q4b_1 = _mm256_set_m128i(_mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_1, 4), m4b)),
                                               _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_1, m4b)));
        const __m256i q4b_2 = _mm256_set_m128i(_mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_2, 4), m4b)),
                                               _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_2, m4b)));
        const __m256i ax_1 = _mm256_sign_epi8(q4b_1, q4b_1);   // mul_add_epi8's |x|, hoisted
        const __m256i ax_2 = _mm256_sign_epi8(q4b_2, q4b_2);
        const float dx_1 = sck_fp16(x[ib + 0].d);
        const float dx_2 = sck_fp16(x[ib + 1].d);
        // ---- APPLY N TIMES ----
        for (int k = 0; k < NACT; ++k) {
            const blk_q8_0 * y = ys[k];
            const __m256i q8b_1 = _mm256_loadu_si256((const __m256i *) y[ib + 0].qs);
            const __m256i q8b_2 = _mm256_loadu_si256((const __m256i *) y[ib + 1].qs);
            const __m256i p16_1 = _mm256_maddubs_epi16(ax_1, _mm256_sign_epi8(q8b_1, q4b_1));
            const __m256i p16_2 = _mm256_maddubs_epi16(ax_2, _mm256_sign_epi8(q8b_2, q4b_2));
            const __m256i p_1 = _mm256_madd_epi16(p16_1, mone);
            const __m256i p_2 = _mm256_madd_epi16(p16_2, mone);
            acc1[k] = _mm256_fmadd_ps(_mm256_set1_ps(sck_fp16(y[ib + 0].d) * dx_1), _mm256_cvtepi32_ps(p_1), acc1[k]);
            acc2[k] = _mm256_fmadd_ps(_mm256_set1_ps(sck_fp16(y[ib + 1].d) * dx_2), _mm256_cvtepi32_ps(p_2), acc2[k]);
        }
    }
    for (int k = 0; k < NACT; ++k) outs[k] = sck_hsum_float_8(_mm256_add_ps(acc1[k], acc2[k]));
}

}  // namespace sck
