#!/usr/bin/env bash
set -euo pipefail

# Execute the Falnama notebooks from the repository root.
# Usage:
#   ./scripts/run_notebooks.sh
#   ./scripts/run_notebooks.sh pipeline/01_market_selector.ipynb pipeline/02_anomaly_detector.ipynb

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ "$#" -gt 0 ]]; then
  NOTEBOOKS=("$@")
else
  NOTEBOOKS=(
    "pipeline/01_market_selector.ipynb"
    "pipeline/02_anomaly_detector.ipynb"
    "pipeline/03_ai_communicator.ipynb"
    "pipeline/04_trading_paper_recommendations.ipynb"
  )
fi

for notebook in "${NOTEBOOKS[@]}"; do
  if [[ ! -f "${notebook}" ]]; then
    echo "Missing notebook: ${notebook}" >&2
    exit 1
  fi
  echo "Executing ${notebook}"
  jupyter nbconvert --execute --to notebook --inplace "${notebook}"
done
