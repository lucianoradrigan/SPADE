/*
 * Flash linker map for SPADE models on the LAUNCHXL-F28379D (CPU1). It differs from C2000Ware's
 * 2837xD_FLASH_lnk_cpu1.cmd in two merged regions, because a model needs contiguous memory:
 *   ARENA      RAMGS0..RAMGS15 for the tensors (131 056 bytes)
 *   FLASHDATA  sectors F..J for the weights and reference windows (320 KB)
 * The last 8 words of RAM and flash stay reserved (errata "Prefetching Beyond Valid Memory").
 */
MEMORY
{
PAGE 0 :
   BEGIN      : origin = 0x080000, length = 0x000002
   RAMM0      : origin = 0x000123, length = 0x0002DD
   RAMLS0     : origin = 0x008000, length = 0x002800
   RESET      : origin = 0x3FFFC0, length = 0x000002
   FLASHB     : origin = 0x082000, length = 0x002000
   FLASHCDE   : origin = 0x084000, length = 0x00C000
   FLASHDATA  : origin = 0x090000, length = 0x028000
PAGE 1 :
   BOOT_RSVD  : origin = 0x000002, length = 0x000121
   RAMM1      : origin = 0x000400, length = 0x0003F8
   RAMD0      : origin = 0x00B000, length = 0x001000
   RAMLS5     : origin = 0x00A800, length = 0x000800
   ARENA      : origin = 0x00C000, length = 0x00FFF8
}

SECTIONS
{
   codestart    : > BEGIN      PAGE = 0, ALIGN(8)
   .cinit       : > FLASHB     PAGE = 0, ALIGN(8)
   .init_array  : > FLASHB     PAGE = 0, ALIGN(8)
   .switch      : > FLASHB     PAGE = 0, ALIGN(8)
   .text        : > FLASHCDE   PAGE = 0, ALIGN(8)
   .const       : > FLASHDATA  PAGE = 0, ALIGN(8)
   .reset       : > RESET,     PAGE = 0, TYPE = DSECT
   .stack       : > RAMM1      PAGE = 1
   .bss         : > RAMD0      PAGE = 1
   .bss:output  : > RAMD0      PAGE = 1
   .bss:cio     : > RAMD0      PAGE = 1
   .data        : > RAMD0      PAGE = 1
   .sysmem      : > RAMD0      PAGE = 1
   .spade_arena : > ARENA      PAGE = 1

   .TI.ramfunc  : {} LOAD = FLASHCDE,
                     RUN = RAMLS0,
                     LOAD_START(RamfuncsLoadStart),
                     LOAD_SIZE(RamfuncsLoadSize),
                     LOAD_END(RamfuncsLoadEnd),
                     RUN_START(RamfuncsRunStart),
                     RUN_SIZE(RamfuncsRunSize),
                     RUN_END(RamfuncsRunEnd),
                     PAGE = 0, ALIGN(8)
}
