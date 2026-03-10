#!/usr/bin/env bash
# Run retrieval_score for all model families
set -e
SCRIPT="src/retrieval_score/retrieval_score.py"

for family in Cerebras-GPT OpenAI Pythia Qwen2.5 Qwen3-Embedding; do
    echo "========================================"
    echo "  $family"
    echo "========================================"
    python "$SCRIPT" --model_family "$family" --abtt 2
done

echo ""
echo "All done. Results under: out/retrieval_score/"
