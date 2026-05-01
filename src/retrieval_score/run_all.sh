#!/usr/bin/env bash
# Run retrieval_score for all model families.
# Optional environment variables:
#   EMBEDDING_LEVEL=chunk  # book or chunk, default book
#   CHUNK_SAMPLE_N=5000    # chunk mode sample size, default 5000
#   SAMPLE_SEED=42         # chunk mode sample seed, default 42
set -e
SCRIPT="src/retrieval_score/retrieval_score.py"
embedding_level="${EMBEDDING_LEVEL:-book}"
chunk_sample_n="${CHUNK_SAMPLE_N:-5000}"
sample_seed="${SAMPLE_SEED:-42}"

for family in Cerebras-GPT OpenAI Pythia Qwen2.5 Qwen3-Embedding; do
    echo "========================================"
    echo "  $family"
    echo "========================================"
    if [[ "$embedding_level" == "chunk" ]]; then
        python "$SCRIPT" \
            --model_family "$family" \
            --embedding_level chunk \
            --sample_n "$chunk_sample_n" \
            --sample_seed "$sample_seed"
    else
        python "$SCRIPT" --model_family "$family"
    fi
done

echo ""
echo "All done. Results under: out/retrieval_score/"
