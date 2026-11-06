#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export PATH="$HOME/.local/bin:$PATH"
python make_tables.py && python make_figures.py && tectonic main.tex
