#!/usr/bin/env bash
# One-command regeneration of every publication_package/ number and figure.
#   pip install -r requirements.txt && npm install -g ganache
#   bash publication_package/run_all_experiments.sh
# Optional: PLANTUML_JAR=/path/to/plantuml-1.2024.7.jar to re-render the diagrams.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"
exec "$PYTHON" publication_package/run_all_experiments.py "$@"
