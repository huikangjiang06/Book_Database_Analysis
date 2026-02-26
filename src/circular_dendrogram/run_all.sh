#!/bin/bash

export ANTHROPIC_API_KEY=""

declare -A model_sizes
model_sizes["Cerebras-GPT"]="111M 256M 590M 1.3B 2.7B 6.7B 13B"
model_sizes["OpenAI"]="text-embedding-3-small text-embedding-3-large"
model_sizes["Pythia"]="70M 160M 410M 1B 1.4B 2.8B 6.9B 12B"
model_sizes["Qwen2.5"]="0.5B 1.5B 3B 7B"
model_sizes["Qwen3-Embedding"]="0.6B 4B 8B"

CLAUDE_MODEL="claude-sonnet-4-5-20250929"

for family in "${!model_sizes[@]}"; do
    for size in ${model_sizes[$family]}; do
        python src/circular_dendrogram/cluster.py --model_family "$family" --model_size "$size" || continue
        python src/circular_dendrogram/topics.py --model_family "$family" --model_size "$size" || continue
        python src/circular_dendrogram/llm_topics.py --model_family "$family" --model_size "$size" --model "$CLAUDE_MODEL" || continue
        python src/circular_dendrogram/visualize.py --model_family "$family" --model_size "$size" || continue
    done
done