#!/bin/zsh
set -euo pipefail

project_dir="${0:A:h}"
runtime_dir="${SECONDEYE_RUNTIME_DIR:-${HOME}/Library/Caches/SecondEye/venv}"
executable="${runtime_dir}/bin/secondeye"

if [[ ! -x "${executable}" ]]; then
  echo "Chưa có runtime. Chạy trước: ${project_dir}/setup_mvp.sh" >&2
  exit 2
fi

cd "${project_dir}"
exec "${executable}" demo "$@"
