#include <math.h>
#include "selftest.h"
#include "spade_model.h"
#include "test_vectors.h"

static int32_t argmax(const float *v, int32_t n)
{
    int32_t i, top = 0;

    for (i = 1; i < n; i++)
        if (v[i] > v[top])
            top = i;
    return top;
}

void spade_check_output(int32_t vector, spade_check *r)
{
    const float *expected = &spade_test_expected[vector * SPADE_OUTPUT_LEN];
    const float *out = spade_model_output();
    int32_t i;

    r->max_abs_err = 0.0f;
    for (i = 0; i < SPADE_OUTPUT_LEN; i++) {
        float d = fabsf(out[i] - expected[i]);

        if (d > r->max_abs_err)
            r->max_abs_err = d;
    }
    r->top = SPADE_NUM_CLASSES ? argmax(out, SPADE_OUTPUT_LEN) : -1;
    r->expected_top = SPADE_NUM_CLASSES ? argmax(expected, SPADE_OUTPUT_LEN) : -1;
}
