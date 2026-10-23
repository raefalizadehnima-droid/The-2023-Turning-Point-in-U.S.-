#!/usr/bin/env bash
set -euo pipefail
python3 main.py
python3 manuscript_content.py
NODE_PATH=$(npm root -g) node make_docx.js
python3 finalize.py
