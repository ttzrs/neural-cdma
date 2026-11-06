#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
# Python with matplotlib is required. PYTHON allows an explicit project environment.
PYTHON="${PYTHON:-python}"
TECTONIC="${TECTONIC:-$HOME/.local/bin/tectonic}"
if [[ ! -x "$TECTONIC" ]]; then
    TECTONIC="$(command -v tectonic)"
fi
"$PYTHON" make_tables.py
"$PYTHON" make_figures.py
"$TECTONIC" --keep-logs main.tex
"$PYTHON" - <<'PY'
from pathlib import Path
import re
import shutil
import subprocess
log = Path('main.log').read_text(errors='replace')
patterns = [r'There were undefined', r'(?:Citation|Reference) .* undefined',
            r'Warning:.*undefined', r'Rerun to get cross-references right']
if any(re.search(pattern, log, re.IGNORECASE) for pattern in patterns):
    raise SystemExit('Unresolved references in final TeX log')
if shutil.which('pdftotext'):
    text = subprocess.check_output(['pdftotext', 'main.pdf', '-'], text=True)
    if '??' in text:
        raise SystemExit('Unresolved ?? in rendered PDF')
    print('Checked final PDF: no ??; final TeX log: no unresolved references.')
else:
    raise SystemExit('pdftotext is required to verify the PDF contains no ??')
PY
