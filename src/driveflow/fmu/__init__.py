"""Command line for the OpenModelica FMUs of SPADE (round PLECS/OpenModelica, task B8).

``python -m driveflow.fmu fetch`` would download prebuilt FMUs from a GitHub Release. It is
disabled: the licence review (docs/audit/plecs_om/b8_license_review.md) did not clear the
redistribution of the OpenModelica runtime's third-party parts, so no Release exists. Build the FMUs
locally instead (docs/guides/BUILD_FMUS_LOCALLY.md)."""

FETCH_DISABLED = (
    "Prebuilt FMU download is disabled: the licence review of the OpenModelica runtime inside the FMUs "
    "(docs/audit/plecs_om/b8_license_review.md) did not clear their redistribution, so no Release is published. "
    "Build them locally: scripts/fmu/build_fmus.sh (see docs/guides/BUILD_FMUS_LOCALLY.md)."
)
