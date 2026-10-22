#!/usr/bin/env bash
# Runs omc from the pinned OpenModelica Docker image with the repository mounted at /work and a
# persistent OpenModelica home (MSL installed once) at data/fmu/openmodelica_home.
#   scripts/fmu/om.sh scripts/fmu/some_script.mos
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
# shellcheck disable=SC1091
source <(grep -E '^(OM_IMAGE|OM_IMAGE_DIGEST|MSL_VERSION)=' "$here/OPENMODELICA_VERSION")
data_dir="$(readlink -f "${SPADE_DATA_DIR:-$repo/data}")"   # SPADE_DATA_DIR: use another checkout's data/ (git worktrees)
mkdir -p "$data_dir/fmu"
home_dir="$data_dir/fmu/openmodelica_home"
mkdir -p "$home_dir"
image="${OM_IMAGE%%:*}@${OM_IMAGE_DIGEST}"
if [ ! -d "$home_dir/libraries/Modelica ${MSL_VERSION}" ]; then
  docker run --rm -u "$(id -u):$(id -g)" -e HOME=/home/om -v "$home_dir:/home/om/.openmodelica" "$image" \
    bash -lc "printf 'updatePackageIndex(); installPackage(Modelica, \"${MSL_VERSION}\", exactMatch=true); getErrorString();\n' > /tmp/i.mos && omc /tmp/i.mos"
fi
exec docker run --rm -u "$(id -u):$(id -g)" -e HOME=/home/om -v "$home_dir:/home/om/.openmodelica" -v "$repo:/work" -v "$data_dir:/work/data" -w /work "$image" omc "$@"
