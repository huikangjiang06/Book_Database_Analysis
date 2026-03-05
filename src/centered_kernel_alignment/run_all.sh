#!/usr/bin/env bash
# Run CKA analysis for all model families.
# Must be executed from the project root:
#   bash src/centered_kernel_alignment/run_all.sh

set -euo pipefail

cd "$(dirname "$0")/../.."   # ensure we are always at project root

families=("Cerebras-GPT" "OpenAI" "Pythia" "Qwen2.5" "Qwen3-Embedding")

for family in "${families[@]}"; do
    echo "========================================"
    echo "Running CKA for: $family"
    echo "========================================"
    python src/centered_kernel_alignment/centered_kernel_alignment.py \
        --model_family "$family"
    echo ""
done

echo "All done. Results under: out/centered_kernel_alignment/"
