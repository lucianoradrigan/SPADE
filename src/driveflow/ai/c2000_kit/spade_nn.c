/*
 * spade_nn.c - see spade_nn.h. The conv kernels run from RAM on the C28x: flash wait states
 * dominate their cost otherwise.
 */
#include <math.h>
#include "spade_nn.h"

#ifdef __TI_COMPILER_VERSION__
#pragma CODE_SECTION(spade_conv1d, ".TI.ramfunc")
#pragma CODE_SECTION(spade_depthwise1d, ".TI.ramfunc")
#endif

void spade_conv1d(const float *in, int32_t steps, int32_t in_ch, float *out, int32_t filters,
                  const float *kernel, const float *bias, int32_t k, int32_t dilation, int32_t pad_left)
{
    int32_t t, f, kk, c;

    for (t = 0; t < steps; t++) {
        for (f = 0; f < filters; f++) {
            float acc = bias[f];

            for (kk = 0; kk < k; kk++) {
                int32_t src = t + kk * dilation - pad_left;

                if (src < 0 || src >= steps)
                    continue;
                for (c = 0; c < in_ch; c++)
                    acc += in[src * in_ch + c] * kernel[(kk * in_ch + c) * filters + f];
            }
            out[t * filters + f] = acc;
        }
    }
}

void spade_depthwise1d(const float *in, int32_t steps, int32_t in_ch, float *out, int32_t mult,
                       const float *kernel, const float *bias, int32_t k, int32_t dilation, int32_t pad_left)
{
    int32_t t, c, m, kk;

    for (t = 0; t < steps; t++) {
        for (c = 0; c < in_ch; c++) {
            for (m = 0; m < mult; m++) {
                float acc = bias[c * mult + m];

                for (kk = 0; kk < k; kk++) {
                    int32_t src = t + kk * dilation - pad_left;

                    if (src >= 0 && src < steps)
                        acc += in[src * in_ch + c] * kernel[(kk * in_ch + c) * mult + m];
                }
                out[(t * in_ch + c) * mult + m] = acc;
            }
        }
    }
}

void spade_dense(const float *in, int32_t in_n, float *out, int32_t units, const float *kernel, const float *bias)
{
    int32_t u, i;

    for (u = 0; u < units; u++) {
        float acc = bias[u];

        for (i = 0; i < in_n; i++)
            acc += in[i] * kernel[i * units + u];
        out[u] = acc;
    }
}

/* Batch-normalization at inference is a per-channel scale + offset, folded at export time. */
void spade_batchnorm(const float *in, int32_t steps, int32_t ch, float *out, const float *scale, const float *offset)
{
    int32_t t, c;

    for (t = 0; t < steps; t++)
        for (c = 0; c < ch; c++)
            out[t * ch + c] = in[t * ch + c] * scale[c] + offset[c];
}

void spade_relu(const float *in, int32_t n, float *out)
{
    int32_t i;

    for (i = 0; i < n; i++)
        out[i] = in[i] > 0.0f ? in[i] : 0.0f;
}

void spade_sigmoid(const float *in, int32_t n, float *out)
{
    int32_t i;

    for (i = 0; i < n; i++)
        out[i] = 1.0f / (1.0f + expf(-in[i]));
}

void spade_softmax(const float *in, int32_t n, float *out)
{
    int32_t i;
    float top = in[0], sum = 0.0f;

    for (i = 1; i < n; i++)
        if (in[i] > top)
            top = in[i];
    for (i = 0; i < n; i++) {
        out[i] = expf(in[i] - top);
        sum += out[i];
    }
    for (i = 0; i < n; i++)
        out[i] /= sum;
}

void spade_gap1d(const float *in, int32_t steps, int32_t ch, float *out)
{
    int32_t t, c;

    for (c = 0; c < ch; c++) {
        double acc = 0.0;

        for (t = 0; t < steps; t++)
            acc += in[t * ch + c];
        out[c] = (float)(acc / steps);
    }
}

void spade_maxpool1d(const float *in, int32_t steps, int32_t ch, float *out, int32_t out_steps,
                     int32_t pool, int32_t stride, int32_t pad_left)
{
    int32_t o, c, p;

    for (o = 0; o < out_steps; o++) {
        for (c = 0; c < ch; c++) {
            int32_t first = 1;
            float top = 0.0f;

            for (p = 0; p < pool; p++) {
                int32_t src = o * stride + p - pad_left;

                if (src < 0 || src >= steps)
                    continue;
                if (first || in[src * ch + c] > top)
                    top = in[src * ch + c];
                first = 0;
            }
            out[o * ch + c] = top;
        }
    }
}

/* b_steps == 1 broadcasts b over time (the squeeze-and-excitation gate). */
void spade_mul(const float *a, const float *b, int32_t steps, int32_t ch, int32_t b_steps, float *out)
{
    int32_t t, c;

    for (t = 0; t < steps; t++)
        for (c = 0; c < ch; c++)
            out[t * ch + c] = a[t * ch + c] * b[(b_steps == 1 ? 0 : t) * ch + c];
}

void spade_add(const float *a, const float *b, int32_t n, float *out)
{
    int32_t i;

    for (i = 0; i < n; i++)
        out[i] = a[i] + b[i];
}

void spade_normalize(const float *raw, int32_t steps, int32_t ch, const float *level_center,
                     const float *level_scale, float *out, float *mu, float *sigma)
{
    int32_t t, c;

    for (c = 0; c < ch; c++) {
        double sum = 0.0, var = 0.0;
        float m, s;

        if (level_scale[c] > 0.0f) {
            mu[c] = level_center[c];
            sigma[c] = level_scale[c];
            for (t = 0; t < steps; t++)
                out[t * ch + c] = (raw[t * ch + c] - level_center[c]) / level_scale[c];
            continue;
        }
        for (t = 0; t < steps; t++)
            sum += raw[t * ch + c];
        sum /= steps;
        for (t = 0; t < steps; t++) {
            double d = raw[t * ch + c] - sum;

            var += d * d;
        }
        m = (float)sum;
        s = (float)sqrt(var / steps);
        mu[c] = m;
        if (s > 1e-3f * (fabsf(m) + 1e-6f)) {
            sigma[c] = s;
            for (t = 0; t < steps; t++)
                out[t * ch + c] = (raw[t * ch + c] - m) / s;
        } else {
            sigma[c] = 1.0f;
            for (t = 0; t < steps; t++)
                out[t * ch + c] = 0.0f;
        }
    }
}

int32_t spade_decide(float *probs, int32_t n, float temperature, float abstain_below)
{
    int32_t i, top = 0;

    if (temperature > 0.0f) {
        float sum = 0.0f;

        for (i = 0; i < n; i++) {
            probs[i] = powf(probs[i], 1.0f / temperature);
            sum += probs[i];
        }
        for (i = 0; i < n; i++)
            probs[i] /= sum;
    }
    for (i = 1; i < n; i++)
        if (probs[i] > probs[top])
            top = i;
    return probs[top] < abstain_below ? -1 : top;
}
