# Book Embedding Analysis

This repo is for studying semantic representations of 500 books across multiple language model families and scales.

## Model Families

Embeddings generated for multiple model sizes within each family:

- **Qwen3-Embedding**: 0.6B, 4B, 8B
- **Qwen2.5**: 0.5B, 1.5B, 3B, 7B
- **Pythia**: 70M, 160M, 410M, 1B, 1.4B, 2.8B, 6.9B, 12B
- **Cerebras-GPT**: 111M, 256M, 590M, 1.3B, 2.7B, 6.7B, 13B
- **OpenAI**: text-embedding-3-small, text-embedding-3-large

**Note**: Pythia and Cerebras-GPT families train all models on the same data in the same sequence, providing training checkpoints for each model size. This enables analyzing how much training is needed for models to develop specific capabilities.

## Directory Structure

```
outputs_embeddings_all_with_chunks/        # From Naz
├── processed_books/          
├── <ModelFamily>/
│   └── <size>/
│       └── <genre>/
│           └── *.pkl
└── manifest_*.jsonl          

out/
├── <family>/<size>/          # Clustering results
├── centered_kernel_alignment/<family>/
├── gutenberg_meta/
├── HDBSCAN_stability_score/<family>/
├── main_components_removal/        # Some of the outputs are not uploaded due to size
├── retrieval_score/<family>/
└── stability_across_model_size/<family>/

src/        # Analyses Methods
├── circular_dendrogram/
├── centered_kernel_alignment/
├── retrieval_score/
├── HDBSCAN_stability_score/
├── knn_community/
├── stability_across_model_size/
└── main_components_removal/  

utils/      # Wrapper around anthropic api for LLM queries
├── api.py
```

## Analyses

### 1. Clustering & Visualization
```bash
bash src/circular_dendrogram/run_all.sh  # Run pipeline for all model families and all model sizes
```

**Pipeline**: UMAP → HDBSCAN → LLM Topic Extraction → Interactive HTML visualizer

**Outputs** (`out/<family>/<size>/`):
- `clusters.pkl` - Cluster assignments and embeddings
- `cluster_topics.json` - c-TF-IDF keywords per cluster
- `per-book-relation/book_explorer.html` - Interactive visualization
- `summary.json` - Clustering statistics

### 2. Gutenberg Meta Extraction

Project Gutenberg provides metadata through the Gutendex API (https://gutendex.com/) with more granular labels beyond genre. Key fields include **Subject** (e.g., "Movie Books", "Horror Fiction", "Monsters") and **Bookshelves** (e.g., "British Literature", "Science Fiction by Women"), enabling additional retrieval and categorization tasks.

Download metadata from gutendex api:
```bash
python extract_gutenberg_meta.py
```

Calculuate metadata stats:
```bash
python genre_stats.py
python gutenberg_stats.py
```

**Outputs** (`out/gutenberg_meta/`):
- `meta.jsonl`: full metadata
- `bookshelf_counts.json`: bookshelf field counts
- `genre_counts.json`: genre field counts
- `subject_count.json`: subject field counts

### 3. All-But-The-Top (ABTT) Main Component Removal

Embedding spaces are often "cone-shaped" with most embeddings clustered in one cone. This can be verified by calculating mean cosine similarity (e.g., 0.999355 for Pythia 70M, 0.84134 for Cerebras-GPT 111M). ABTT mitigates this by mean-centering embeddings and projecting out the top-N principal directions via PCA, revealing phenomena hidden by over-centered embeddings.

Calculate and cache average cosine similarity, average embedding, and principal components for every model family and model size:
```bash
python average_cosine_similarity.py
python average_embedding.py
python pca_embedding.py
```

Post-process embeddings by mean-centering and removing top N principal components:
```python
from src.main_components_removal.ABTT import abtt

embeddings = ...  # Load embeddings for a specific model family and size

processed_embedding = abtt(
    family="Qwen3-Embedding",
    size="4B",
    embeddings=embeddings,
    n=5,
)
```

**Outputs** (out/main_components_removal/):
- `avg_cos_sim.json` - Average cosine similarity per model
- `avg_embed.pkl` - Average embedding vector per model
- `eigenvalues_top_20.json` - Top 20 eigenvalues from PCA per model
- `eigenvalues.json` - All eigenvalues from PCA per model
- `principal_directions.pkl` - PCA principal directions per model

### 4. Centered Kernel Alignment (CKA)

Compares distributions of representations across model sizes by: (1) building kernel matrices (500×500 pairwise similarity), (2) centering by subtracting row/column means and adding grand mean, (3) computing ⟨A,B⟩/√(⟨A,A⟩⟨B,B⟩) using Frobenius inner product. Measures how similar embedding geometries are across scales.

```bash
python src/centered_kernel_alignment/centered_kernel_alignment.py \
    --model_family Qwen3-Embedding

bash src/centered_kernel_alignment/run_all.sh  # Run for all model families
```

Measures representation similarity between model sizes using linear, cosine, and RBF kernels.

**Outputs** (`out/centered_kernel_alignment/<family>/`):
- `results.json` - CKA matrices for all kernel types
- `heatmap.png` - Visualization of cross-model similarity
- `bar.png` - Bar plot visualization

### 5. Retrieval Score

Evaluates emergent ability through a retrieval task: for each book, select another book sharing some genre/subject/bookshelf, plus N unrelated books. Measures how often the model correctly retrieves the related book from the distractor set, testing if embeddings capture categorical structure.

```bash
python src/retrieval_score/retrieval_score.py \
    --model_family Qwen3-Embedding \
    --num_instances 200 \
    --num_candidates 19 \
    --abtt 5  # Optionally apply ABTT with top 5 components removed

bash src/retrieval_score/run_all.sh  # Run for all model families
```

Tests embedding quality via category-based retrieval (genre, bookshelf, subject).

**Outputs** (`out/retrieval_score/<family>/`):
- `results.json` - Accuracy per grouping and model size
- `retrieval_score.png` - Performance curves across scales

### 6. HDBSCAN Stability Score

HDBSCAN provides a stability score for each cluster measuring the cleanliness of cluster boundaries. Analyzes these persistence scores across model sizes to assess whether larger models produce more stable, well-defined book communities.

```bash
python src/HDBSCAN_stability_score/HDBSCAN_stability_score.py \
    --model_family Qwen3-Embedding

bash src/HDBSCAN_stability_score/run_all.sh  # Run for all model families
```

Analyzes cluster persistence scores across model sizes.

**Outputs** (`out/HDBSCAN_stability_score/<family>/`):
- `results.json` - Per-cluster stability statistics
- `hdbscan_stability.png` - Stability trends

### 7. Cross-Scale Neighborhood Stability

Measures embedding space divergence between models by: (1) ranking all 499 books by distance to each query book, (2) computing Spearman rank correlation between models, (3) calculating Jaccard similarity of top-k neighbor sets. Shows how much the embedding geometry changes across model scales.

```bash
python src/stability_across_model_size/stability_across_model_size.py \
    --model_family Qwen3-Embedding \
    --k 15  # Number of nearest neighbours for Jaccard Similarity (default: 10)
```

Measures how book neighborhoods change between adjacent model sizes (Spearman ρ, Jaccard similarity).

**Outputs** (`out/stability_across_model_size/<family>/`):
- `results.json` - Per-book stability scores
- `stability.png` - Cross-scale consistency
- `heatmap.png` - Heatmap visualization

## Quick Start
1. Download outputs_embeddings_all_with_chunks and place it in the root directory.
2. Run BERTopic workflow for all models:
```bash
bash src/circular_dendrogram/run_all.sh
```
3. Run metadata extraction and stats:
```bash
python extract_gutenberg_meta.py
python genre_stats.py
python gutenberg_stats.py
```
4. Run average cosine similarity, embedding and PCA for ABTT:
```bash
python src/main_components_removal/average_cosine_similarity.py
python src/main_components_removal/average_embedding.py
python src/main_components_removal/pca_embedding.py
```
5. Run anything else.