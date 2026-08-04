#!/usr/bin/env bash
set -euo pipefail

python3 -m py_compile evaluate.py model_registry.py
python3 -m unittest discover -s tests -v
python3 evaluate.py --write_template /tmp/selective_revision_schema.jsonl
python3 - <<'PY'
import json
from pathlib import Path
path = Path('/tmp/selective_revision_schema.jsonl')
with path.open(encoding='utf-8') as handle:
    for line in handle:
        json.loads(line)
print('schema template: valid JSONL')
PY
