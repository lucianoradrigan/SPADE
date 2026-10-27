# Third-party notices for SPADE's OpenModelica FMUs

These texts go with every FMU that SPADE redistributes (the GitHub Release built by `.github/workflows/fmu-release.yml`). The FMUs are built with OpenModelica 1.27.1 (MSL 4.1.0+maint.om, CVODE). The Windows and macOS FMUs (X1) are the Model Exchange build of the same model compiled from the FMU's C sources on that OS, without CVODE. An exported FMU carries no licence file of its own, so the notices are attached here. The review behind this list is `docs/audit/plecs_om/b8_license_review.md`.

| Component (where in the FMU) | Licence | File |
|---|---|---|
| SPADE's Modelica models (generated code) | MIT | `SPADE_MIT_LICENSE.txt` |
| OpenModelica Run-Time System (`binaries/`, `sources/util`, `simulation`, `meta`, `gc`, `fmi-export`, …) | BSD New, chosen among BSD New / AGPL v3 / OSMC-PL 1.8 as the header allows | `OPENMODELICA_RUNTIME_BSD-3-Clause.txt`, `OPENMODELICA_RUNTIME_LICENSE_HEADER.txt` |
| SUNDIALS 5.4.0 CVODE (`binaries/linux64/libsundials_*.so`, `sources/sundials`) | BSD-3-Clause (LLNS / SMU) | `SUNDIALS_5.4.0_LICENSE.txt`, `SUNDIALS_5.4.0_NOTICE.txt` |
| FMI 2.0 headers (`sources/fmi`) | BSD-2-Clause (MODELISAR / Modelica Association) | `FMI_2.0_HEADERS_BSD-2-Clause.txt` |
| Modelica Standard Library C sources (`sources/ModelicaExternalC`) | BSD-3-Clause (Modelica Association) | `MODELICA_STANDARD_LIBRARY_C_BSD-3-Clause.txt`, `MODELICA_MATIO_BSD-3-Clause.txt` |
| `snprintf.c` (P. Powell, H. Weiss) | permissive notice | `SNPRINTF_NOTICE.txt` |
| LAPACK subset via CLAPACK 3.2.1 (OpenModelica `3rdParty/dgesv`; compiled into the binary) | modified BSD (University of Tennessee) | `CLAPACK_3.2.1_COPYING.txt` |
| libf2c support routines (same subset) | AT&T / Lucent / Bellcore permission notice | `LIBF2C_NOTICE.txt` |
| MINPACK via cminpack (OpenModelica `3rdParty/CMinpack`; compiled into the binary) | Minpack Copyright Notice (1999), University of Chicago | `MINPACK_COPYRIGHT.txt` |
| `sources/util/libcsv.h` (header only; libcsv code is not compiled into the FMU) | LGPL-2.1 (libcsv) | `LGPL-2.1.txt` |
| Windows FMU only (X1, compiled with MinGW-w64 GCC and `--static-runtime`): libgcc | GPL-3.0 with the GCC Runtime Library Exception (no notice required for the compiled output) | — |
| Windows FMU only: mingw-w64 winpthreads, linked statically into the DLL | MIT-style (mingw-w64 project) and BSD-style (Lockless Inc.) | `MINGW_W64_WINPTHREADS_COPYING.txt` |
| `sources/config.guess`, `config.sub` (build scripts, not linked) | GPL-3.0 with the Autoconf exception (text inside each file) | — |

**Acknowledgement required by MINPACK:** This product includes software developed by the University of Chicago, as Operator of Argonne National Laboratory.
