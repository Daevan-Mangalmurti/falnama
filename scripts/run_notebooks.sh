#!/usr/bin/env bash
set -euo pipefail

# Execute Jupyter notebooks in-place
# Usage: ./scripts/run_notebooks.sh

jupyter nbconvert --execute --to notebook --inplace 01_market_selector.ipynb
jupyter nbconvert --execute --to notebook --inplace 02_anomaly_detector.ipynb
jupyter nbconvert --execute --to notebook --inplace 03_ai_communicator.ipynb
jupyter nbconvert --execute --to notebook --inplace 04_trading_paper_recommendations.ipynb
