/* selftest.h - replays the exported reference windows through the C model. */
#ifndef SPADE_SELFTEST_H
#define SPADE_SELFTEST_H

#include <stdint.h>

/* Worst absolute difference against the Keras output, and the top-1 of both (-1 for a forecaster). */
typedef struct {
    float max_abs_err;
    int32_t top, expected_top;
} spade_check;

void spade_check_output(int32_t vector, spade_check *r);

#endif
