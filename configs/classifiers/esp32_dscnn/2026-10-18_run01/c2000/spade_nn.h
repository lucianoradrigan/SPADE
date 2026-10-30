/*
 * spade_nn.h - float32 inference kernels for SPADE models exported to the C2000 (C28x).
 *
 * C28x: char and int are 16 bits and sizeof counts 16-bit words, so every index here is int32_t
 * and nothing assumes byte addressing. Tensors are flat float arrays in Keras (T, C) order.
 */
#ifndef SPADE_NN_H
#define SPADE_NN_H

#include <stdint.h>

void spade_conv1d(const float *in, int32_t steps, int32_t in_ch, float *out, int32_t filters,
                  const float *kernel, const float *bias, int32_t k, int32_t dilation, int32_t pad_left);
void spade_depthwise1d(const float *in, int32_t steps, int32_t in_ch, float *out, int32_t mult,
                       const float *kernel, const float *bias, int32_t k, int32_t dilation, int32_t pad_left);
void spade_dense(const float *in, int32_t in_n, float *out, int32_t units, const float *kernel, const float *bias);
void spade_batchnorm(const float *in, int32_t steps, int32_t ch, float *out, const float *scale, const float *offset);
void spade_relu(const float *in, int32_t n, float *out);
void spade_sigmoid(const float *in, int32_t n, float *out);
void spade_softmax(const float *in, int32_t n, float *out);
void spade_gap1d(const float *in, int32_t steps, int32_t ch, float *out);
void spade_maxpool1d(const float *in, int32_t steps, int32_t ch, float *out, int32_t out_steps,
                     int32_t pool, int32_t stride, int32_t pad_left);
void spade_mul(const float *a, const float *b, int32_t steps, int32_t ch, int32_t b_steps, float *out);
void spade_add(const float *a, const float *b, int32_t n, float *out);

/* The training pipeline's per-window rule (windowing.normalize_window_like_training): raw is
 * (steps, ch); level_scale[c] > 0 keeps that channel's absolute level. mu and sigma receive what a
 * forecast needs to return to physical units: y * sigma + mu. */
void spade_normalize(const float *raw, int32_t steps, int32_t ch, const float *level_center,
                     const float *level_scale, float *out, float *mu, float *sigma);

/* metrics.json "calibration": p^(1/T) renormalized, the top class, and -1 (uncertain) below
 * abstain_below. temperature <= 0 means uncalibrated. */
int32_t spade_decide(float *probs, int32_t n, float temperature, float abstain_below);

#endif
