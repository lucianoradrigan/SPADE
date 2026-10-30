/*
 * board.h - LAUNCHXL-F28379D helpers on top of C2000Ware driverlib.
 *
 * Board facts (from C2000Ware device.h with _LAUNCHXL_F28379D):
 *   - 10 MHz crystal -> PLL -> SYSCLK 200 MHz, LSPCLK 50 MHz
 *   - LED1 = GPIO31 (blue, LD10), LED2 = GPIO34 (red, LD9), active low
 *   - SCI-A on GPIO42 (TX) / GPIO43 (RX), bridged to the XDS100v2
 *     virtual COM port (/dev/ttyUSB0)
 */
#ifndef BOARD_H
#define BOARD_H

#include "driverlib.h"
#include "device.h"

#define LED_ON  0U
#define LED_OFF 1U

void board_init(void);              /* clocks, GPIO, PIE, LEDs off */
void sci_init(uint32_t baud);       /* SCI-A 8N1 with FIFO */
void sci_puts(const char *s);       /* '\n' -> "\r\n" */
void sci_putu(uint32_t v);

#endif
