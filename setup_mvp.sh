#!/bin/zsh
set -euo pipefail

project_dir="${0:A:h}"
runtime_dir="${SECONDEYE_RUNTIME_DIR:-${HOME}/Library/Caches/SecondEye/venv}"
python_command="${SECONDEYE_PYTHON:-python3.11}"
cache_dir="${SECONDEYE_CACHE_DIR:-${runtime_dir}/cache}"

export MPLCONFIGDIR="${SECONDEYE_MPLCONFIG_DIR:-${cache_dir}/matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

"${python_command}" -m venv "${runtime_dir}"
"${runtime_dir}/bin/python" -m pip install --upgrade pip setuptools wheel
# The runtime belongs to this checkout. An editable install keeps the CLI in
# sync with source updates instead of silently running an older wheel snapshot.
"${runtime_dir}/bin/python" -m pip install --editable \
  "${project_dir}[dev,detection,multimodal,ocr]"

vision_source="${project_dir}/src/secondeye/multimodal/apple_vision_ocr.m"
vision_binary="${runtime_dir}/bin/secondeye-vision-ocr"
if [[ "$(uname -s)" == "Darwin" ]] && command -v xcrun >/dev/null 2>&1; then
  if ! xcrun clang -fobjc-arc -framework Foundation -framework AppKit \
    -framework Vision "${vision_source}" -o "${vision_binary}"; then
    echo "Cảnh báo: không biên dịch được Apple Vision OCR; sẽ dùng PaddleOCR." >&2
  fi
fi

"${runtime_dir}/bin/secondeye" doctor

echo "SecondEye MVP runtime: ${runtime_dir}"
echo "Activate: source ${runtime_dir}/bin/activate"
echo "Run: ${project_dir}/run_mvp.sh --camera 0"
