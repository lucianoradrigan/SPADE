/*
 * main.c - LAUNCHXL-F28379D benchmark: replays the reference windows, reports timing and parity
 * over SCI-A (115200 8N1) and repeats, so a monitor can attach at any moment. Touches only the
 * blue LED and SCI-A; it drives no PWM or analog output.
 */
#include "board.h"
#include "selftest.h"
#include "spade_model.h"
#include "test_vectors.h"

static uint32_t float_bits(float f)
{
    union { float f; uint32_t u; } v;

    v.f = f;
    return v.u;
}

static void sci_kv(const char *key, uint32_t value)
{
    sci_puts(key);
    sci_putu(value);
}

void main(void)
{
    board_init();
    sci_init(115200);
    CPUTimer_stopTimer(CPUTIMER1_BASE);
    CPUTimer_setPeriod(CPUTIMER1_BASE, 0xFFFFFFFFUL);
    CPUTimer_setPreScaler(CPUTIMER1_BASE, 0U);
    CPUTimer_reloadTimerCounter(CPUTIMER1_BASE);
    CPUTimer_startTimer(CPUTIMER1_BASE);

    for (;;) {
        int32_t v;

        sci_puts("\nSPADE-BEGIN model=" SPADE_MODEL_ID);
        sci_kv(" vectors=", SPADE_TEST_VECTORS);
        sci_kv(" sysclk=", DEVICE_SYSCLK_FREQ);
        sci_kv(" arena_bytes=", SPADE_ARENA_FLOATS * 4UL);
        sci_puts("\n");
        for (v = 0; v < SPADE_TEST_VECTORS; v++) {
            spade_check r;
            uint32_t t0, t1;

            t0 = CPUTimer_getTimerCount(CPUTIMER1_BASE);
            spade_model_run(&spade_test_raw[v * SPADE_INPUT_LEN]);
            t1 = CPUTimer_getTimerCount(CPUTIMER1_BASE);
            spade_check_output(v, &r);
            sci_kv("V ", (uint32_t)v);
            sci_kv(" cycles=", t0 - t1);
            sci_kv(" err_bits=", float_bits(r.max_abs_err));
            sci_kv(" top=", (uint32_t)(r.top + 1));
            sci_kv(" exp=", (uint32_t)(r.expected_top + 1));
            sci_puts("\n");
            GPIO_togglePin(DEVICE_GPIO_PIN_LED1);
        }
        sci_puts("SPADE-END\n");
        DEVICE_DELAY_US(2000000);
    }
}
