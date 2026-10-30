# SPADE model on the LAUNCHXL-F28379D (CPU1, flash, EABI).
#   make          build build/spade_model.out
#   make flash    program it and leave it running
#   make host     same self-test on the PC with gcc
CGT       ?= $(HOME)/ti/ti-cgt-c2000_25.11.1.LTS
UNIFLASH  ?= $(HOME)/ti/uniflash
C2000WARE ?= $(HOME)/ti/src/c2000ware-core-sdk
CCXML     ?= $(HOME)/post_doc/C2000/TMS320F28379D.ccxml
CL        := $(CGT)/bin/cl2000
DLIB      := $(C2000WARE)/driverlib/f2837xd/driverlib
DEVSUP    := $(C2000WARE)/device_support/f2837xd/common

CFLAGS  := -v28 -ml -mt --abi=eabi --cla_support=cla1 --float_support=fpu32 \
           --tmu_support=tmu0 --vcu_support=vcu2 -O2 --opt_for_speed=5 \
           --define=CPU1 --define=_FLASH --define=_LAUNCHXL_F28379D \
           --diag_warning=225 --diag_suppress=10063 --advice:performance=none \
           -I$(CGT)/include -I$(DLIB) -I$(DEVSUP)/include -I.
LDFLAGS := -z --reread_libs --rom_model --entry_point=code_start --stack_size=0x300 \
           --heap_size=0 --warn_sections -i$(CGT)/lib -llibc.a \
           $(DLIB)/ccs/Debug/driverlib_eabi.lib lnk_spade_f28379d.cmd
SOURCES := main.c board.c selftest.c spade_nn.c spade_model.c \
           $(DEVSUP)/source/device.c $(DEVSUP)/source/F2837xD_CodeStartBranch.asm
DSLITE  := $(UNIFLASH)/deskdb/content/TICloudAgent/linux/ccs_base/DebugServer/bin/DSLite

build/spade_model.out: $(wildcard *.c *.h) lnk_spade_f28379d.cmd Makefile
	@mkdir -p build
	$(CL) $(CFLAGS) --obj_directory=build $(SOURCES) $(LDFLAGS) -m build/spade_model.map -o $@

flash: build/spade_model.out
	$(DSLITE) load -c $(CCXML) -n 0 -f $<

host:
	@mkdir -p build
	gcc -O2 -ffp-contract=off -I. host_main.c selftest.c spade_nn.c spade_model.c -lm -o build/host_selftest
	./build/host_selftest

clean:
	rm -rf build

.PHONY: flash host clean
