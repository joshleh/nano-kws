/*
 * Hand-written depthwise-separable convolution kernels (see header).
 *
 * SIMD versions target AVX2 (256-bit, 8 floats) + FMA. Compiler flags
 * are set in CMakeLists.txt to enable the right ISA. There is no
 * runtime CPU dispatch — at the small problem sizes this microbench
 * targets, the overhead would dominate; either build with /arch:AVX2
 * (MSVC) or -mavx2 -mfma (GCC/Clang) or use the naive path only.
 */

#include "conv_kernels.h"

#include <immintrin.h>
#include <stddef.h>

/* ------------------------------------------------------------------ */
/* Pointwise (1x1) conv                                                */
/* ------------------------------------------------------------------ */

/*
 * Naive: triple nested loop. Inner dot product over c_in. No tiling,
 * no register reuse — the compiler's auto-vectoriser will not save us
 * here because the access pattern stripes across input channels for
 * every (oc, p) pair.
 */
void conv_pointwise_naive(
    const float *input,
    const float *weight,
    float       *output,
    int          c_in,
    int          c_out,
    int          hw)
{
    for (int oc = 0; oc < c_out; ++oc) {
        const float *w_row = weight + (size_t)oc * (size_t)c_in;
        float       *o_row = output + (size_t)oc * (size_t)hw;
        for (int p = 0; p < hw; ++p) {
            float sum = 0.0f;
            for (int ic = 0; ic < c_in; ++ic) {
                sum += w_row[ic] * input[(size_t)ic * (size_t)hw + (size_t)p];
            }
            o_row[p] = sum;
        }
    }
}

/*
 * AVX2: for each output channel, walk the spatial axis in chunks of 8
 * floats. Inner loop accumulates contributions from each input channel
 * via _mm256_fmadd_ps (one fused multiply-add per cycle on a modern
 * core). The weight broadcast (`_mm256_set1_ps`) is hoisted outside
 * the spatial loop, so the inner kernel is one load + one FMA per
 * input channel per chunk of 8 outputs.
 *
 * Asymptotic FMA rate ceiling: 1 FMA / cycle * 8 lanes = 8 FLOPs / cycle.
 * Real-world we'll see 2-4× ATen's MKL-DNN kernel because we're missing
 * register tiling, blocking for L1, and prefetching.
 */
void conv_pointwise_avx2(
    const float *input,
    const float *weight,
    float       *output,
    int          c_in,
    int          c_out,
    int          hw)
{
    const int p_simd_end = (hw / 8) * 8;

    for (int oc = 0; oc < c_out; ++oc) {
        const float *w_row = weight + (size_t)oc * (size_t)c_in;
        float       *o_row = output + (size_t)oc * (size_t)hw;

        /* SIMD body: 8 spatial outputs at a time. */
        for (int p = 0; p < p_simd_end; p += 8) {
            __m256 acc = _mm256_setzero_ps();
            for (int ic = 0; ic < c_in; ++ic) {
                __m256 w = _mm256_set1_ps(w_row[ic]);
                __m256 x = _mm256_loadu_ps(input + (size_t)ic * (size_t)hw + (size_t)p);
                acc = _mm256_fmadd_ps(w, x, acc);
            }
            _mm256_storeu_ps(o_row + p, acc);
        }

        /* Scalar tail for the last <8 outputs. */
        for (int p = p_simd_end; p < hw; ++p) {
            float sum = 0.0f;
            for (int ic = 0; ic < c_in; ++ic) {
                sum += w_row[ic] * input[(size_t)ic * (size_t)hw + (size_t)p];
            }
            o_row[p] = sum;
        }
    }
}

/* ------------------------------------------------------------------ */
/* Depthwise 3x3 conv (stride=1, padding=1)                           */
/* ------------------------------------------------------------------ */

/*
 * Compute one output element with explicit zero-padding boundary checks.
 * Used by the scalar implementation and by the SIMD implementation to
 * handle the edge columns / top + bottom rows where the SIMD path would
 * either read out-of-bounds or pull in padding values.
 */
static inline float depthwise_3x3_at(
    const float *in_ch,
    const float *k,
    int          h,
    int          w,
    int          oh,
    int          ow)
{
    float sum = 0.0f;
    for (int kh = 0; kh < 3; ++kh) {
        int ih = oh + kh - 1;
        if (ih < 0 || ih >= h) continue;
        for (int kw = 0; kw < 3; ++kw) {
            int iw = ow + kw - 1;
            if (iw < 0 || iw >= w) continue;
            sum += in_ch[(size_t)ih * (size_t)w + (size_t)iw] * k[kh * 3 + kw];
        }
    }
    return sum;
}

void conv_depthwise_3x3_naive(
    const float *input,
    const float *weight,
    float       *output,
    int          c,
    int          h,
    int          w)
{
    for (int ch = 0; ch < c; ++ch) {
        const float *in_ch  = input  + (size_t)ch * (size_t)h * (size_t)w;
        const float *k      = weight + (size_t)ch * 9;
        float       *out_ch = output + (size_t)ch * (size_t)h * (size_t)w;
        for (int oh = 0; oh < h; ++oh) {
            for (int ow = 0; ow < w; ++ow) {
                out_ch[(size_t)oh * (size_t)w + (size_t)ow] =
                    depthwise_3x3_at(in_ch, k, h, w, oh, ow);
            }
        }
    }
}

/*
 * AVX2 depthwise 3x3:
 *
 * For each channel, broadcast the 9 kernel weights into 9 vector
 * registers. Then for each interior row (1 <= oh <= h-2), process the
 * interior columns in chunks of 8 outputs. Each chunk loads 3 rows of
 * (chunk_width + 2) input floats and does 9 unaligned loads + 9 FMAs.
 *
 * Top row, bottom row, left column, right column, and the SIMD tail
 * fall back to the scalar `depthwise_3x3_at` so we don't have to write
 * special-case AVX2 prologue/epilogue. This keeps the SIMD kernel
 * simple (one branch-free hot loop) at the cost of leaving a thin
 * border in the scalar path.
 */
void conv_depthwise_3x3_avx2(
    const float *input,
    const float *weight,
    float       *output,
    int          c,
    int          h,
    int          w)
{
    if (h < 1 || w < 1) return;

    for (int ch = 0; ch < c; ++ch) {
        const float *in_ch  = input  + (size_t)ch * (size_t)h * (size_t)w;
        const float *k      = weight + (size_t)ch * 9;
        float       *out_ch = output + (size_t)ch * (size_t)h * (size_t)w;

        const __m256 k00 = _mm256_set1_ps(k[0]);
        const __m256 k01 = _mm256_set1_ps(k[1]);
        const __m256 k02 = _mm256_set1_ps(k[2]);
        const __m256 k10 = _mm256_set1_ps(k[3]);
        const __m256 k11 = _mm256_set1_ps(k[4]);
        const __m256 k12 = _mm256_set1_ps(k[5]);
        const __m256 k20 = _mm256_set1_ps(k[6]);
        const __m256 k21 = _mm256_set1_ps(k[7]);
        const __m256 k22 = _mm256_set1_ps(k[8]);

        /* Top row (oh=0): scalar (top boundary). */
        for (int ow = 0; ow < w; ++ow) {
            out_ch[(size_t)0 * (size_t)w + (size_t)ow] =
                depthwise_3x3_at(in_ch, k, h, w, 0, ow);
        }

        /* Interior rows. */
        for (int oh = 1; oh < h - 1; ++oh) {
            /* Left edge column (ow=0): scalar. */
            out_ch[(size_t)oh * (size_t)w + 0] =
                depthwise_3x3_at(in_ch, k, h, w, oh, 0);

            /* SIMD interior: process 8 outputs at a time, starting at ow=1.
             * Loop while the rightmost lane (ow + 7) is still strictly
             * inside the right edge column (i.e. ow + 7 < w - 1, so the
             * last full SIMD output is at column w - 2).
             */
            int ow = 1;
            for (; ow + 7 < w - 1; ow += 8) {
                const float *r0 = in_ch + (size_t)(oh - 1) * (size_t)w + (size_t)(ow - 1);
                const float *r1 = in_ch + (size_t)(oh    ) * (size_t)w + (size_t)(ow - 1);
                const float *r2 = in_ch + (size_t)(oh + 1) * (size_t)w + (size_t)(ow - 1);

                __m256 acc = _mm256_setzero_ps();
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(r0 + 0), k00, acc);
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(r0 + 1), k01, acc);
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(r0 + 2), k02, acc);
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + 0), k10, acc);
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + 1), k11, acc);
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + 2), k12, acc);
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + 0), k20, acc);
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + 1), k21, acc);
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + 2), k22, acc);
                _mm256_storeu_ps(out_ch + (size_t)oh * (size_t)w + (size_t)ow, acc);
            }

            /* Scalar tail: any remaining interior columns + the right edge. */
            for (; ow < w; ++ow) {
                out_ch[(size_t)oh * (size_t)w + (size_t)ow] =
                    depthwise_3x3_at(in_ch, k, h, w, oh, ow);
            }
        }

        /* Bottom row (oh=h-1): scalar (bottom boundary). Only when h>1
         * to avoid double-writing the top row at h==1. */
        if (h > 1) {
            for (int ow = 0; ow < w; ++ow) {
                out_ch[(size_t)(h - 1) * (size_t)w + (size_t)ow] =
                    depthwise_3x3_at(in_ch, k, h, w, h - 1, ow);
            }
        }
    }
}
