#!/usr/bin/env bash
# Run mutual k-NN alignment for all model families.
# Must be executed from the project root:
#   bash src/mutual_knn_alignment/run_all.sh
#
# Optional environment variables:
#   K=20                # nearest neighbors, default 10
#   CHUNK_SAMPLE_N=5000 # sampled chunk embeddings per family, default 5000
#   SAMPLE_SEED=42      # random seed for chunk sampling, default 42
#   RUN_ABTT=1          # also run legacy book-level ABTT outputs, default 0

set -euo pipefail

cd "$(dirname "$0")/../.."

SCRIPT="src/mutual_knn_alignment/mutual_knn_alignment.py"
REFERENCE_SCRIPT="src/mutual_knn_alignment/cerebras_13b_comparison.py"
families=("Cerebras-GPT" "OpenAI" "Pythia" "Qwen2.5" "Qwen3-Embedding")
k="${K:-10}"
chunk_sample_n="${CHUNK_SAMPLE_N:-5000}"
sample_seed="${SAMPLE_SEED:-42}"
run_abtt="${RUN_ABTT:-0}"
abtt_n="${ABTT_N:-2}"

for family in "${families[@]}"; do
    echo "========================================"
    echo "Running mutual k-NN for: $family"
    echo "========================================"
    python "$SCRIPT" --model_family "$family" --k "$k"
    echo ""
done

for family in "${families[@]}"; do
    echo "========================================"
    echo "Running chunk mutual k-NN for: $family"
    echo "========================================"
    python "$SCRIPT" \
        --model_family "$family" \
        --k "$k" \
        --embedding_level chunk \
        --sample_n "$chunk_sample_n" \
        --sample_seed "$sample_seed"
    echo ""
done

echo "========================================"
echo "Running all-model comparison to Cerebras-GPT/13B"
echo "========================================"
python "$REFERENCE_SCRIPT" --k "$k"
echo ""

if [[ "$run_abtt" == "1" ]]; then
    for family in "${families[@]}"; do
        echo "========================================"
        echo "Running book mutual k-NN with ABTT n=$abtt_n for: $family"
        echo "========================================"
        python "$SCRIPT" --model_family "$family" --k "$k" --abtt "$abtt_n"
        echo ""
    done

    echo "========================================"
    echo "Running all-model comparison to Cerebras-GPT/13B with ABTT n=$abtt_n"
    echo "========================================"
    python "$REFERENCE_SCRIPT" --k "$k" --abtt "$abtt_n"
    echo ""
fi

echo "All done. Results under: out/mutual_knn_alignment/"
