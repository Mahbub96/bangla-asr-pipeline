#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export COPYFILE_DISABLE=1
export PYTORCH_ENABLE_MPS_FALLBACK=1
export HF_HOME="${HF_HOME:-$PWD/models}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$PWD/models}"

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi

if command -v uv >/dev/null 2>&1; then
  uv pip install --python .venv/bin/python -r requirements_mac_mps.txt
else
  .venv/bin/python -m pip install -r requirements_mac_mps.txt
fi

# Some external macOS volumes create AppleDouble files even inside .venv; these
# can break Transformers' import scanner, so clean them after dependency sync.
find .venv -name '._*' -type f -delete

.venv/bin/python - <<'PY'
import torch
print(f"torch={torch.__version__}")
print(f"mps_built={torch.backends.mps.is_built() if hasattr(torch.backends, 'mps') else False}")
print(f"mps_available={torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False}")
if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
    raise SystemExit('MPS is not available in this Python environment')
PY
