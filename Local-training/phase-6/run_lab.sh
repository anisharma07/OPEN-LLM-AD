#!/bin/bash
# Launch the Phase 6 Arm-B Lab at http://localhost:${PORT:-8765}
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
PY="$HERE/../.venv/bin/python"
"$PY" -c "import fastapi, uvicorn, multipart" 2>/dev/null || {
  echo "Installing lab web dependencies into Local-training/.venv ..."
  if command -v uv >/dev/null; then uv pip install --python "$PY" -r requirements-lab.txt; else "$PY" -m pip install -r requirements-lab.txt; fi
}
echo "Arm-B Lab → http://localhost:${PORT:-8765}"
exec "$PY" -m uvicorn lab.server:app --host "${HOST:-127.0.0.1}" --port "${PORT:-8765}"
