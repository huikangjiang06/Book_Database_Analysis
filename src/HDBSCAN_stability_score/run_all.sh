#!/usr/bin/env bash
# Run HDBSCAN stability score analysis for all model families.
# Must be executed from the project root:
#   bash src/HDBSCAN_stability_score/run_all.sh

set -euo pipefail

cd "$(dirname "$0")/../.."   # ensure we are always at project root

families=("Cerebras-GPT" "OpenAI" "Pythia" "Qwen2.5" "Qwen3-Embedding")

for family in "${families[@]}"; do
    echo "========================================"
    echo "Running for: $family"
    echo "========================================"
    python src/HDBSCAN_stability_score/HDBSCAN_stability_score.py \
        --model_family "$family"
    echo ""
done

echo "All done. Results under: out/HDBSCAN_stability_score/"
