# B8 — Licence review of the OpenModelica FMUs (stop point)

**Date:** 2026-10-25.
**Decision G4, step 1:** confirm the licence of the OpenModelica runtime inside the FMUs before any workflow or Release.

**Result (first pass): not clear enough to redistribute; B8 stopped. Superseded by §5 (resolution after checking the OpenModelica 1.27.1 source): redistribution permitted with the notices in `licenses/fmu/`.**
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

## 5. Resolution — 2026-10-25, user option (b): OpenModelica 1.27.1 source checked

**Commands (sources in a scratch folder, not committed):**

```
git clone --depth 1 --branch v1.27.1 --filter=blob:none --sparse https://github.com/OpenModelica/OpenModelica.git   # 4a7248c1ab64
git clone --filter=blob:none https://github.com/OpenModelica/OMCompiler-3rdParty.git && git checkout dfc66f9ee9ab   # the submodule commit of v1.27.1
grep -rlI -iE "Lesser General Public|LGPL|GNU General Public License" OMCompiler/SimulationRuntime/c
curl -sSL https://www.netlib.org/clapack/clapack.tgz   # SHA-256 6dc4c382164b…; CLAPACK-3.2.1/COPYING
```

**The open points of §2 and §3 are settled:**

| Open point | Finding in the 1.27.1 sources | Licence |
|---|---|---|
| MINPACK / cminpack in the binary | `3rdParty/CMinpack` (git subtree of devernay/cminpack), with `CopyrightMINPACK.txt` | Minpack Copyright Notice (1999), University of Chicago. Redistribution in binary form is allowed with the notice and the acknowledgement "This product includes software developed by the University of Chicago, as Operator of Argonne National Laboratory" |
| LAPACK / BLAS via f2c in the binary | `3rdParty/dgesv`. Its README: "based on clapack (3.2.1), but the f2c header has been modified" | CLAPACK 3.2.1 `COPYING`: modified BSD, University of Tennessee. libf2c `Notice` (netlib `f2c/libf2c.zip`): AT&T / Lucent / Bellcore permission notice |
| "Some third-party code in the runtime is LGPL" | The only LGPL file in `SimulationRuntime/c` is `util/libcsv.c` (LGPL-2.1). `util/jni*.h` are GPL-2.0 with the Classpath exception. The FMUs contain neither `libcsv.c` nor the JNI headers. They contain only `sources/util/libcsv.h`, an 88-line header, and the binary has no `csv_` symbol | LGPL-2.1 text attached for that header |
| Boehm GC | not compiled in: the `GC_asprintf` / `GC_vasprintf` symbols are OpenModelica's own (`util/modelica_string.c`, runtime licence) | — |

**Licence of the OpenModelica runtime.** It is confirmed by `OSMC-Runtime-License.txt` in the v1.27.1 tree: BSD New, AGPL v3 or OSMC-PL 1.8, at the recipient's choice. SPADE chooses BSD New.

**Verdict: redistribution is permitted, with the notices.** Every component is permissive (BSD-2, BSD-3, the MINPACK and libf2c notices). The one LGPL file in the FMU is a header in `sources/`, distributed with the LGPL-2.1 text. The texts are in `licenses/fmu/` (index: `licenses/fmu/README.md`). The release workflow attaches them as `THIRD_PARTY_NOTICES.txt` and `licenses.zip`, and the MINPACK acknowledgement goes in the Release notes.

**B8 continues as specified:**
- workflow `fmu-release.yml`: Linux required, a Windows attempt, no macOS;
- a draft Release with only B3-GO FMUs, minus DcMotorAveraged (G2);
- `python -m driveflow.fmu fetch` and the Plant Studio button (a job).

**Publishing the Release still needs your confirmation.** This review is not a legal opinion.

## 6. Draft release built by CI (2026-10-25)

- **Trigger:** tag `fmu-v1` on `dev` (`1e49522`). `workflow_dispatch` is not possible while `fmu-release.yml` is not on the default branch.
- **Run:** [40230235712](https://github.com/lucianoradrigan/SPADE/actions/runs/40230235712). The workflow concluded `success`:

| Job | Result |
|---|---|
| `linux` | success: build, B3 on that build, release assets |
| `draft-release` | success |
| `windows-attempt` | failure, as allowed (`continue-on-error`). Chocolatey has OpenModelica up to 1.22.2, not the pinned 1.27.1, so no Windows FMU is built. |
| macOS | not built: there is no official OpenModelica macOS release |

- **Release `fmu-v1`:** `isDraft: true`, **not published**. Assets:

| Asset | Content |
|---|---|
| `BuckAveraged-linux64.fmu` | SHA-256 `013dacb29c5ab7c6…`, B3 GO on the CI build |
| `fmu_manifest.json` | `.mo` `ec5e4545…`, commit `1e49522`. `not_attached`: BuckSwitching and DcMotorSwitching (NO-GO on that build), DcMotorAveraged (G2) |
| `THIRD_PARTY_NOTICES.txt`, `licenses.zip` | the licence texts of `licenses/fmu/` |

- **Check of the assets:**
  - `gh release download fmu-v1` gives the FMU SHA-256 equal to the manifest.
  - The assets were served from a local HTTP server and installed with `SPADE_FMU_RELEASE_URL=… python -m driveflow.fmu fetch --release fmu-v1`: SHA-256 checked, installed.
  - `SPADE/OmBuck-v0` on that FMU, with the native PI's actions, matches `SPADE/Buck-v0` to 6.0e-6 NRMS on v_out.
- **Note:** `fetch` against GitHub works only after the release is published; a draft is not publicly downloadable.

**Publishing the draft needs the user's confirmation.**

## 7. Published (2026-10-26, at the user's request)

- **Command:** `gh release edit fmu-v1 --draft=false --latest`. Release: https://github.com/lucianoradrigan/SPADE/releases/tag/fmu-v1, the latest published release.
- **Model check before publishing:** `modelica/SPADE/package.mo` on `dev` still has SHA-256 `ec5e4545…`, the one in `fmu_manifest.json`, so `fetch` installs the FMU without a model-version warning.
- **Check after publishing:** `python -m driveflow.fmu fetch --dest <scratch>`, with no tag, resolves the latest release `fmu-v1`:
  - it installed `BuckAveraged.fmu`, SHA-256 `013dacb29c5a…`, B3 GO;
  - `SPADE/OmBuck-v0` on that file, under the native PI's actions, matches `SPADE/Buck-v0` to 6.0e-6 NRMS on v_out.
