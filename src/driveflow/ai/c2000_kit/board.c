/*
 * board.c - LAUNCHXL-F28379D bring-up: clocks, LEDs, SCI-A.
 */
#include "board.h"

static void led_init(uint32_t pin)
{
    GPIO_setMasterCore(pin, GPIO_CORE_CPU1);
    GPIO_setPadConfig(pin, GPIO_PIN_TYPE_STD);
    GPIO_writePin(pin, LED_OFF);
    GPIO_setDirectionMode(pin, GPIO_DIR_MODE_OUT);
}

static void sci_pin_init(uint32_t pin, uint32_t cfg, GPIO_Direction dir)
{
    GPIO_setMasterCore(pin, GPIO_CORE_CPU1);
    GPIO_setPinConfig(cfg);
    GPIO_setDirectionMode(pin, dir);
    GPIO_setPadConfig(pin, GPIO_PIN_TYPE_STD);
    GPIO_setQualificationMode(pin, GPIO_QUAL_ASYNC);
}

void board_init(void)
{
    Device_init();                  /* WDT off, PLL 200 MHz, flash setup */
    Device_initGPIO();
    Interrupt_initModule();
    Interrupt_initVectorTable();

    led_init(DEVICE_GPIO_PIN_LED1);
    led_init(DEVICE_GPIO_PIN_LED2);
}

void sci_init(uint32_t baud)
{
    sci_pin_init(DEVICE_GPIO_PIN_SCIRXDA, DEVICE_GPIO_CFG_SCIRXDA,
                 GPIO_DIR_MODE_IN);
    sci_pin_init(DEVICE_GPIO_PIN_SCITXDA, DEVICE_GPIO_CFG_SCITXDA,
                 GPIO_DIR_MODE_OUT);

    SCI_performSoftwareReset(SCIA_BASE);
    SCI_setConfig(SCIA_BASE, DEVICE_LSPCLK_FREQ, baud,
                  SCI_CONFIG_WLEN_8 | SCI_CONFIG_STOP_ONE |
                  SCI_CONFIG_PAR_NONE);
    SCI_resetChannels(SCIA_BASE);
    SCI_resetRxFIFO(SCIA_BASE);
    SCI_resetTxFIFO(SCIA_BASE);
    SCI_clearInterruptStatus(SCIA_BASE, SCI_INT_TXFF | SCI_INT_RXFF);
    SCI_enableFIFO(SCIA_BASE);
    SCI_enableModule(SCIA_BASE);
    SCI_performSoftwareReset(SCIA_BASE);
}

void sci_puts(const char *s)
{
    while (*s) {
        if (*s == '\n')
            SCI_writeCharBlockingFIFO(SCIA_BASE, '\r');
        SCI_writeCharBlockingFIFO(SCIA_BASE, (uint16_t)*s++);
    }
}

void sci_putu(uint32_t v)
{
    char buf[11];
    int i = 0;

    do {
        buf[i++] = '0' + (char)(v % 10U);
        v /= 10U;
    } while (v);
    while (i)
        SCI_writeCharBlockingFIFO(SCIA_BASE, (uint16_t)buf[--i]);
}
