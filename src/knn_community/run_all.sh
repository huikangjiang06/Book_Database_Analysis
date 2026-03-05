#!/usr/bin/env bash
# Run kNN community structure analysis for all model families.
# Must be executed from the project root:
#   bash src/knn_community/run_all.sh

set -euo pipefail

cd "$(dirname "$0")/../.."   # ensure we are always at project root

families=("Cerebras-GPT" "OpenAI" "Pythia" "Qwen2.5" "Qwen3-Embedding")

# Sweep parameters — adjust for a lighter/heavier run
K_MIN=5
K_MAX=80
N_K=8
GAMMA_MIN=0.5
GAMMA_MAX=2.0
N_GAMMA=7
N_SEEDS=3
N_EIGS=30
GAMMA_REF=1.0

for family in "${families[@]}"; do
    echo "========================================"
    echo "Running kNN community: $family"
    echo "========================================"
    python src/knn_community/knn_community.py \
        --model_family "$family" \
        --k_min       "$K_MIN"  \
        --k_max       "$K_MAX"  \
        --n_k         "$N_K"    \
        --gamma_min   "$GAMMA_MIN" \
        --gamma_max   "$GAMMA_MAX" \
        --n_gamma     "$N_GAMMA"   \
        --n_seeds     "$N_SEEDS"   \
        --n_eigs      "$N_EIGS"    \
        --gamma_ref   "$GAMMA_REF"
    echo ""
done

echo "All done. Results under: out/knn_community/"
