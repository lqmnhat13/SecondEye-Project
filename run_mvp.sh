#!/bin/zsh
set -euo pipefail

project_dir="${0:A:h}"
runtime_dir="${SECONDEYE_RUNTIME_DIR:-${HOME}/Library/Caches/SecondEye/venv}"
executable="${runtime_dir}/bin/secondeye"
cache_dir="${SECONDEYE_CACHE_DIR:-${runtime_dir}/cache}"

if [[ ! -x "${executable}" ]]; then
  echo "Chưa có runtime. Chạy trước: ${project_dir}/setup_mvp.sh" >&2
  exit 2
fi

export PYTHONPATH="${project_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${SECONDEYE_MPLCONFIG_DIR:-${cache_dir}/matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

cd "${project_dir}"
exec "${executable}" demo "$@"
