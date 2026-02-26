families=("Cerebras-GPT" "OpenAI" "Pythia" "Qwen2.5" "Qwen3-Embedding")
k=20

for faily in "${families[@]}"; do 
    echo "Running for $faily"
    python src/stability_across_model_size/stability_across_model_size.py --model_family $faily --k $k
done