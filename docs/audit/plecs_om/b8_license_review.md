# B8 — Licence review of the OpenModelica FMUs (stop point)

**Date:** 2026-10-25.
**Decision G4, step 1:** confirm the licence of the OpenModelica runtime inside the FMUs before any workflow or Release.

**Result: not clear enough to redistribute. B8 stops here, as the round's instructions require.**
- No workflow `fmu-release.yml` and no Release (not even a draft) were created.
- B8 is reduced to the local-build guide (`docs/guides/BUILD_FMUS_LOCALLY.md`) and a `python -m driveflow.fmu fetch` that is disabled and says why.
- This is not a legal opinion.

## 1. What was inspected

`BuckAveraged.fmu`, the B3 GO model (SHA-256 `48d52b6a…`, OpenModelica 1.27.1, MSL 4.1.0+maint.om). Commands:

```
unzip -l data/fmu/BuckAveraged.fmu
unzip -q data/fmu/BuckAveraged.fmu -d <dir>
grep -rhoI -i "SPDX-License-Identifier…|BSD…|Lawrence Livermore|LGPL|GNU General Public|…" <dir>/sources --include=*.[ch]
ldd <dir>/binaries/linux64/BuckAveraged.so
nm -D --defined-only <dir>/binaries/linux64/BuckAveraged.so
```

| Item | Finding |
|---|---|
| Licence / notice files in the FMU (`documentation/`, `licenses/`, `LICENSE`, `COPYING`, `NOTICE`) | **None**, in any of the four FMUs. modelDescription.xml has no licence or copyright attribute; OpenModelica does not export them (OpenModelica issue #13169). |
| `binaries/linux64/` | `BuckAveraged.so`, `libsundials_cvode.so.5(.4.0)`, `libsundials_nvecserial.so.5(.4.0)`. Dynamic deps: only those two plus libc / libm. |
| Symbols compiled into `BuckAveraged.so` | OpenModelica runtime; CLAPACK / BLAS translated by f2c (`dgesv_`, `dgetrf_`, `dlamch_`, `f2c_dgemm`, …); MINPACK / cminpack (`hybrj_`, `dogleg_`, `qrfac_`, `enorm_`); the runtime's GC hooks (`GC_asprintf`, `mmc_GC_state`). |

## 2. Third-party components and their licences

| Component (where) | Licence as stated in the files | Status |
|---|---|---|
| OpenModelica simulation runtime (`sources/util`, `simulation`, `meta`, `gc`, `fmi-export`, `math-support`, `optimization`, `linearization`, `dataReconciliation`; 144 files) | Header: "BSD NEW LICENSE OR AGPL VERSION 3 OR OSMC-PL 1.8, ACCORDING TO RECIPIENT'S CHOICE". Same as https://openmodelica.org/osmc-pl/osmc-pl-1.8-runtime.txt | Clear: choose BSD New; the notice must go with binaries. |
| SUNDIALS CVODE 5.4.0 (`sources/sundials`, both `.so`) | `SPDX-License-Identifier: BSD-3-Clause`, Lawrence Livermore National Security | Clear; the notice must go with binaries. |
| FMI 2.0 headers (`sources/fmi`) | MODELISAR consortium, 2-Clause BSD | Clear; the notice is required. |
| Modelica Standard Library C sources (`sources/ModelicaExternalC`, `ModelicaMatIO` incl. Christopher C. Hulbert) | BSD-3-Clause (Modelica Association) | Clear; the notice is required. |
| `snprintf.c` (Patrick Powell, Holger Weiss) | Permissive; the notice is kept in source distributions | Clear. |
| **CLAPACK / BLAS via f2c** (`sources/external_solvers/d*.c`, `f2c.h`, `F77_aloc.c`, `s_*.c`, …) | **No licence header in the files.** Upstream LAPACK is modified BSD and libf2c is under a permissive AT&T/Lucent notice, but the exact upstream versions and texts bundled by OpenModelica 1.27.1 were **not confirmed** | Not clear. |
| **MINPACK / cminpack** (`hybrj_.c`, `dogleg_.c`, `qrfac_.c`, …, `minpack.h`, `cminpack.h`) | **No licence header in the files.** Upstream MINPACK and cminpack carry the "Minpack Copyright Notice (1999) University of Chicago", which needs an acknowledgement in the end-user documentation. Which version OpenModelica ships, and its exact text, were **not confirmed** | Not clear. |
| `config.guess`, `config.sub` in `sources/` (build scripts, not linked) | GPL-3.0 with the Autoconf exception | Redistribution is allowed with the exception; the notice is kept. Not linked. |
| SPADE's model `modelica/SPADE/package.mo` | MIT (this repository) | Clear. |

**Official statement found that contradicts "all permissive":**
- An OpenModelica developer wrote on the forum, 2019-12-20 (https://openmodelica.org/forum/default-topic/24-licensing-of-runtime): "The OpenModelica simulation runtime system already has a different license from the rest of OpenModelica… Some third-party code in it is LGPL-licensed though."
- The post does not say which code.
- Neither the 1.27.1 documentation nor the license files that could be reached confirm or rule this out:
  - `OSMC-License.txt` in the OpenModelica repository does not mention the runtime or generated code;
  - the tagged `external_solvers` folder of v1.27.1 returned 404 to an anonymous fetch.
- The source scan above found no LGPL header. The f2c / MINPACK files have no header at all.

## 3. What redistribution would require (if the open points are settled)

1. Choose BSD New for the OpenModelica runtime and say so, e.g. in `OSMC-USAGE-MODE.txt` or the Release notes.
2. Ship next to each FMU the licence texts of:
   - the OpenModelica runtime (BSD New, OSMC copyright);
   - SUNDIALS (BSD-3, LLNS);
   - the FMI headers (BSD-2, MODELISAR);
   - the MSL C sources (BSD-3, Modelica Association);
   - CLAPACK / LAPACK and libf2c;
   - MINPACK / cminpack, with its acknowledgement clause;
   - `snprintf.c`;
   - and SPADE (MIT).
3. Confirm, from the OpenModelica 1.27.1 source tree or from the OpenModelica developers, the provenance and licence of the f2c / MINPACK sources and of the "LGPL third-party code". Rule it out, or meet the LGPL for the binary (e.g. dynamic linking and source offer).

## 4. Decision needed (user)

Redistributing the FMUs as Release assets needs points 2–3 settled. Options:
- **(a) Keep local builds only.** This is the current state; `fetch` stays disabled.
- **(b) Authorise me to pull the OpenModelica 1.27.1 source** (git tag `v1.27.1`, `OMCompiler/SimulationRuntime/c/external_solvers` and `3rdParty`) to settle the f2c / MINPACK / LGPL questions. If it is clean, I go on with B8 as specified: workflow, draft Release with the licence texts, and `fetch`.
- **(c) Ask the OpenModelica developers** (forum or GitHub) which runtime parts are LGPL.

Publishing the Release still needs your confirmation in every case.
