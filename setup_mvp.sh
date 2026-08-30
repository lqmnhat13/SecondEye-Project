#!/bin/zsh
set -euo pipefail

project_dir="${0:A:h}"
runtime_dir="${SECONDEYE_RUNTIME_DIR:-${HOME}/Library/Caches/SecondEye/venv}"
python_command="${SECONDEYE_PYTHON:-python3.11}"

"${python_command}" -m venv "${runtime_dir}"
"${runtime_dir}/bin/python" -m pip install --upgrade pip setuptools wheel
"${runtime_dir}/bin/python" -m pip install "${project_dir}[dev,detection,multimodal,ocr]"
"${runtime_dir}/bin/secondeye" doctor

echo "SecondEye MVP runtime: ${runtime_dir}"
echo "Activate: source ${runtime_dir}/bin/activate"
echo "Run: ${project_dir}/run_mvp.sh --camera 0"
