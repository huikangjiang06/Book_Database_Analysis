"""
cerebras_13b_comparison.py
==========================
Compare every available model embedding space to Cerebras-GPT/13B using the
mutual k-nearest-neighbor alignment metric.

Outputs (under out/mutual_knn_alignment/cerebras_13b_comparison*/):
  results.json  - mean/std/median mNN against Cerebras-GPT/13B
  scatter.png   - cross-family scatter plot

Usage:
    python src/mutual_knn_alignment/cerebras_13b_comparison.py
    python src/mutual_knn_alignment/cerebras_13b_comparison.py --abtt 2
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mutual_knn_alignment import (
    BASE_OUT,
    _size_sort_key,
    align_to_reference,
    cosine_sim_matrix,
    get_sizes,
    knn_indices,
    load_model_data,
    mutual_knn_per_book,
)

FAMILIES = ["Cerebras-GPT", "Pythia", "Qwen2.5", "Qwen3-Embedding", "OpenAI"]
REFERENCE_FAMILY = "Cerebras-GPT"
REFERENCE_SIZE = "13B"

FAMILY_COLORS = {
    "Cerebras-GPT": "#1f77b4",
    "Pythia": "#ff7f0e",
    "Qwen2.5": "#2ca02c",
    "Qwen3-Embedding": "#d62728",
    "OpenAI": "#9467bd",
}


def size_millions(size: str) -> float | None:
    if size.startswith("text-embedding-3-"):
        return None
    value = _size_sort_key(size)
    return None if not np.isfinite(value) else float(value)


def short_size_label(size: str) -> str:
    return (
        size
        .replace("text-embedding-3-", "te3-")
        .replace("Qwen3-Embedding", "Qwen3")
    )


def compute_all(k: int, abtt_n: int = 0) -> dict:
    print(f"[ref] Loading {REFERENCE_FAMILY}/{REFERENCE_SIZE}" +
          (f" with ABTT n={abtt_n}" if abtt_n > 0 else ""))
    ref_emb_all, ref_ids_all = load_model_data(REFERENCE_FAMILY, REFERENCE_SIZE, abtt_n=abtt_n)
    ref_id_set = set(ref_ids_all)
    records = []

    for family in FAMILIES:
        sizes = get_sizes(family)
        print(f"\n[family] {family}: {sizes}")
        for size in sizes:
            print(f"  {size:>25}", end=" ... ", flush=True)
            emb, ids = load_model_data(family, size, abtt_n=abtt_n)
            id_set = set(ids)
            common_ids = [eid for eid in ref_ids_all if eid in id_set and eid in ref_id_set]
            if k < 1 or k >= len(common_ids):
                print(f"skipping, only {len(common_ids)} common books")
                continue
            ref_emb = align_to_reference(ref_emb_all, ref_ids_all, common_ids)
            emb = align_to_reference(emb, ids, common_ids)
            ref_top = knn_indices(cosine_sim_matrix(ref_emb), k)
            top = knn_indices(cosine_sim_matrix(emb), k)
            scores = mutual_knn_per_book(top, ref_top, len(common_ids))

            record = {
                "family": family,
                "size": size,
                "label": f"{family}/{size}",
                "model_size_millions": size_millions(size),
                "n_common_books": len(common_ids),
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores, ddof=0)),
                "median": float(np.median(scores)),
                "per_book": scores.tolist(),
            }
            records.append(record)
            print(f"mean={record['mean']:.4f}")

    return {
        "reference_family": REFERENCE_FAMILY,
        "reference_size": REFERENCE_SIZE,
        "k": k,
        "abtt_n": abtt_n,
        "n_books": len(ref_ids_all),
        "records": records,
    }


def sort_records(records: list[dict]) -> list[dict]:
    family_rank = {family: i for i, family in enumerate(FAMILIES)}
    return sorted(
        records,
        key=lambda r: (
            r["model_size_millions"] is None,
            r["model_size_millions"] if r["model_size_millions"] is not None else _size_sort_key(r["size"]),
            family_rank.get(r["family"], 999),
            r["size"],
        ),
    )


def plot_scatter(results: dict, out_path: str) -> None:
    records = sort_records(results["records"])

    sizes = []
    for record in records:
        if record["size"] not in sizes:
            sizes.append(record["size"])
    x_pos = {size: i for i, size in enumerate(sizes)}
    offsets = {
        family: offset
        for family, offset in zip(FAMILIES, np.linspace(-0.25, 0.25, len(FAMILIES)))
    }

    fig, ax = plt.subplots(figsize=(max(10, len(sizes) * 0.65), 5.2))
    title = f"All Models vs {REFERENCE_FAMILY}/{REFERENCE_SIZE}"
    if results["abtt_n"] > 0:
        title += f" (ABTT n={results['abtt_n']})"
    fig.suptitle(
        f"{title}\nMutual k-NN alignment (k={results['k']}, N={results['n_books']} books)",
        fontsize=12,
    )

    for family in FAMILIES:
        fam_records = [r for r in records if r["family"] == family]
        xs = [x_pos[r["size"]] + offsets[family] for r in fam_records]
        ys = [r["mean"] for r in fam_records]
        ax.scatter(
            xs,
            ys,
            s=60,
            color=FAMILY_COLORS[family],
            edgecolor="white",
            linewidth=0.7,
            label=family,
            alpha=0.9,
        )

    ax.set_xticks(np.arange(len(sizes)))
    ax.set_xticklabels([short_size_label(size) for size in sizes],
                       rotation=35, ha="right", fontsize=9)
    ax.set_xlabel("Model Size")
    ax.set_ylabel(f"mNN with {REFERENCE_FAMILY}/{REFERENCE_SIZE}")
    ax.set_ylim(-0.05, 1.05)
    ax.yaxis.grid(True, linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, ncols=3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved -> {out_path}")


def print_summary(results: dict) -> None:
    records = sort_records(results["records"])
    title = f"All Models vs {REFERENCE_FAMILY}/{REFERENCE_SIZE} (k={results['k']})"
    if results["abtt_n"] > 0:
        title += f" ABTT n={results['abtt_n']}"
    print(f"\n{'='*84}")
    print(f"  {title}")
    print(f"{'='*84}")
    print(f"  {'Model':>34}  {'Mean':>8}  {'Std':>8}  {'Median':>8}")
    print(f"  {'-'*34}  {'-'*8}  {'-'*8}  {'-'*8}")
    for record in records:
        print(f"  {record['label']:>34}  {record['mean']:>8.4f}  "
              f"{record['std']:>8.4f}  {record['median']:>8.4f}")
    print(f"{'='*84}\n")


def strip_per_book(record: dict) -> dict:
    return {k: v for k, v in record.items() if k != "per_book"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare all model sizes to Cerebras-GPT/13B with mutual k-NN"
    )
    parser.add_argument("--k", type=int, default=10,
                        help="Number of nearest neighbors for mNN (default: 10)")
    parser.add_argument("--abtt", type=int, default=0, metavar="N",
                        help="Apply ABTT: remove top-N principal directions (0 = disabled)")
    args = parser.parse_args()

    suffix = "_abtt" if args.abtt > 0 else ""
    out_dir = os.path.join(BASE_OUT, "cerebras_13b_comparison" + suffix)
    json_out = os.path.join(out_dir, "results.json")
    scatter_out = os.path.join(out_dir, "scatter.png")

    results = compute_all(args.k, abtt_n=args.abtt)
    print_summary(results)

    os.makedirs(out_dir, exist_ok=True)
    save = {
        "reference_family": results["reference_family"],
        "reference_size": results["reference_size"],
        "k": results["k"],
        "abtt_n": results["abtt_n"],
        "n_books": results["n_books"],
        "records": [strip_per_book(r) for r in sort_records(results["records"])],
    }
    with open(json_out, "w") as f:
        json.dump(save, f, indent=2)
    print(f"[save] Summary -> {json_out}")

    plot_scatter(results, scatter_out)


if __name__ == "__main__":
    main()
