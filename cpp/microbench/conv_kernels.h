/*
 * Hand-written depthwise-separable convolution kernels.
 *
 * The two ops here are the workhorses of every depthwise-separable
 * convolutional architecture (MobileNet, DS-CNN, EfficientNet, ...):
 *
 *   conv_pointwise_*       1x1 conv = channel mixing = a (C_out x C_in)
 *                          matrix-vector product applied at every spatial
 *                          position. Dominates the FLOPs of DS-CNN.
 *
 *   conv_depthwise_3x3_*   3x3 conv with groups=C = independent per-channel
 *                          spatial filtering. Cheap (one filter per channel)
 *                          but memory-bound.
 *
 * Two implementations are provided for each:
 *   * `_naive` — straight nested loops, no SIMD. The "what would I write
 *     in 5 minutes from scratch" baseline.
 *   * `_avx2`  — AVX2 (256-bit / 8 floats wide) intrinsics with FMA. The
 *     "what does the hand-rolled SIMD path look like" version.
 *
 * Layout: NCHW, fp32, batch size 1. Pointwise pre-flattens spatial into
 * H*W. All buffers are caller-allocated and assumed contiguous.
 *
 * The C ABI is plain `extern "C"` so this library can be loaded from
 * Python via ctypes or from C++ directly.
 */

#ifndef NANO_KWS_CONV_KERNELS_H
#define NANO_KWS_CONV_KERNELS_H

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* Pointwise (1x1) conv                                                */
/* ------------------------------------------------------------------ */

/* input  layout: [c_in, hw]  (channels-first, spatial flattened)       */
/* weight layout: [c_out, c_in]                                         */
/* output layout: [c_out, hw]                                           */
void conv_pointwise_naive(
    const float *input,
    const float *weight,
    float       *output,
    int          c_in,
    int          c_out,
    int          hw);

void conv_pointwise_avx2(
    const float *input,
    const float *weight,
    float       *output,
    int          c_in,
    int          c_out,
    int          hw);

/* ------------------------------------------------------------------ */
/* Depthwise 3x3 conv (stride=1, padding=1)                           */
/* ------------------------------------------------------------------ */

/* input  layout: [c, h, w]                                            */
/* weight layout: [c, 3, 3]                                            */
/* output layout: [c, h, w]                                            */
void conv_depthwise_3x3_naive(
    const float *input,
    const float *weight,
    float       *output,
    int          c,
    int          h,
    int          w);

void conv_depthwise_3x3_avx2(
    const float *input,
    const float *weight,
    float       *output,
    int          c,
    int          h,
    int          w);

#ifdef __cplusplus
}
#endif

#endif /* NANO_KWS_CONV_KERNELS_H */
