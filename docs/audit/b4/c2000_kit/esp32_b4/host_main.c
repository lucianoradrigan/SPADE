/* host_main.c - the same self-test on the PC (gcc), the check that needs no board. */
#include <stdio.h>
#include "selftest.h"
#include "spade_model.h"
#include "test_vectors.h"

int main(void)
{
    int32_t v;

    for (v = 0; v < SPADE_TEST_VECTORS; v++) {
        spade_check r;

        spade_model_run(&spade_test_raw[v * SPADE_INPUT_LEN]);
        spade_check_output(v, &r);
        printf("V %d err=%.9e top=%d exp=%d\n", (int)v, r.max_abs_err, (int)r.top, (int)r.expected_top);
    }
    return 0;
}
