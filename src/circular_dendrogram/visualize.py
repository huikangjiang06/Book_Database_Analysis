"""
Visualization Step
==================
Generates a self-contained interactive HTML book explorer.

For each book (selectable via dropdown):
  Left:  Ego-network — selected book at center, top-K nearest neighbors on a
         ring. Edge width ∝ cosine similarity; node color = cluster ID.
  Right: 2D UMAP scatter — all books (grey), selected book (star) and its
         neighbors (colored) highlighted.

Output:
    out/<model_tag>/per-book-relation/book_explorer.html

Usage:
    python visualize.py --model_family Qwen3-Embedding --model_size 0.6B
    python visualize.py --model_family Pythia --model_size 70M --top_k 8
"""

import argparse
import json
import math
import os
import pickle

import numpy as np

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(ROOT, "out")
BOOK_TOPICS_DIR = os.path.join(OUT_DIR, "Book_Topics")


def _tag(model_family: str, model_size: str) -> str:
    return f"{model_family}_{model_size}".replace("/", "_").replace(" ", "_")


def _run_dir(model_family: str, model_size: str) -> str:
    return os.path.join(OUT_DIR, model_family, model_size)


# ─── Step 1: Load All Data ────────────────────────────────────────────────────

def load_data(model_family: str, model_size: str) -> dict:
    tag = _tag(model_family, model_size)
    run_dir = _run_dir(model_family, model_size)

    clusters_pkl = os.path.join(run_dir, "clusters.pkl")
    cluster_topics_json = os.path.join(run_dir, "cluster_topics.json")

    if not os.path.exists(clusters_pkl):
        raise FileNotFoundError(f"Run pipeline.py first: {clusters_pkl}")
    if not os.path.exists(cluster_topics_json):
        raise FileNotFoundError(f"Run topics.py first: {cluster_topics_json}")

    print(f"[load] Reading {clusters_pkl} ...")
    with open(clusters_pkl, "rb") as f:
        cluster_data = pickle.load(f)

    with open(cluster_topics_json) as f:
        cluster_topics = json.load(f)

    df = cluster_data["result_df"]
    embeddings = cluster_data["embeddings"]   # (N, D) L2-normalised
    umap2 = cluster_data["umap2"]             # (N, 2)
    labels = cluster_data["labels"]           # (N,)

    # Load per-book keywords from Book_Topics/
    print(f"[load] Loading book keyword files ...")
    book_keywords = {}
    for _, row in df.iterrows():
        stem = os.path.splitext(os.path.basename(row["pkl_path"]))[0]
        path = os.path.join(BOOK_TOPICS_DIR, stem + ".json")
        if os.path.exists(path):
            with open(path) as f:
                bt = json.load(f)
            book_keywords[stem] = [kw["word"] for kw in bt.get("top_keywords", [])[:8]]
        else:
            book_keywords[stem] = []

    condensed_tree = cluster_data.get("condensed_tree")  # pd.DataFrame or None

    return {
        "df": df,
        "embeddings": embeddings,
        "umap2": umap2,
        "labels": labels,
        "cluster_topics": cluster_topics,
        "book_keywords": book_keywords,
        "run_dir": run_dir,
        "tag": tag,
        "condensed_tree": condensed_tree,
    }


# ─── Step 2: Compute Nearest Neighbours ───────────────────────────────────────

def compute_neighbors(embeddings: np.ndarray, top_k: int = 10) -> list[list[dict]]:
    """
    For each book compute its top-K nearest neighbours using cosine similarity.
    Since embeddings are already L2-normalised, cosine_sim = dot product.
    Returns: list of lists — neighbors[i] = [{idx, sim}, ...] sorted desc.
    """
    print(f"[neighbors] Computing cosine similarity matrix ({len(embeddings)} × {len(embeddings)}) ...")
    # Full dot-product similarity matrix
    sim_matrix = embeddings @ embeddings.T  # (N, N)
    np.fill_diagonal(sim_matrix, -1.0)       # exclude self

    neighbors = []
    for i in range(len(embeddings)):
        row = sim_matrix[i]
        top_idx = np.argsort(row)[::-1][:top_k]
        neighbors.append([
            {"idx": int(j), "sim": round(float(row[j]), 4)}
            for j in top_idx
        ])
    print(f"[neighbors] Done.")
    return neighbors


# ─── Step 3: Build Serialisable Book Data ─────────────────────────────────────

# Fixed palette — supports up to 30 clusters + noise
_PALETTE = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
    "#a65628", "#f781bf", "#999999", "#66c2a5", "#fc8d62",
    "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f", "#e5c494",
    "#b3b3b3", "#1b9e77", "#d95f02", "#7570b3", "#e7298a",
    "#66a61e", "#e6ab02", "#a6761d", "#666666", "#8dd3c7",
    "#ffffb3", "#bebada", "#fb8072", "#80b1d3", "#fdb462",
]
_NOISE_COLOR = "#cccccc"


def _cluster_color(cid: int) -> str:
    if cid < 0:
        return _NOISE_COLOR
    return _PALETTE[cid % len(_PALETTE)]


def build_book_data(data: dict, neighbors: list[list[dict]]) -> tuple[list, dict, dict]:
    """
    Returns:
        books_json     — list of book dicts (one per book, indexed by position)
        ct_json        — cluster_topics summary {cluster_id_str: [word, ...]}
        cluster_labels — LLM-generated label per cluster {cluster_id_str: label_str}
    """
    df = data["df"]
    umap2 = data["umap2"]
    labels = data["labels"]
    book_keywords = data["book_keywords"]

    books_json = []
    for i, (_, row) in enumerate(df.iterrows()):
        stem = os.path.splitext(os.path.basename(row["pkl_path"]))[0]
        cid = int(labels[i])
        books_json.append({
            "id": i,
            "title": row["book_title"],
            "genre": row["genre"],
            "cluster": cid,
            "color": _cluster_color(cid),
            "umap_x": round(float(umap2[i, 0]), 4),
            "umap_y": round(float(umap2[i, 1]), 4),
            "keywords": book_keywords.get(stem, []),
            "neighbors": neighbors[i],
        })

    # Cluster topics: {cluster_id_str: [word, ...]}
    ct_json = {}
    for key, val in data["cluster_topics"].items():
        ct_json[key] = [kw["word"] for kw in val.get("top_keywords", [])[:6]]

    # LLM labels: {cluster_id_str: label_str}  (only those that have one)
    cluster_labels = {
        key: val["llm_label"]
        for key, val in data["cluster_topics"].items()
        if val.get("llm_label", "")
    }

    return books_json, ct_json, cluster_labels


# ─── Step 4: Build Cluster Legend Data ────────────────────────────────────────

def build_cluster_legend(labels: np.ndarray) -> list[dict]:
    from collections import Counter
    counts = Counter(int(l) for l in labels)
    legend = []
    for cid in sorted(counts.keys()):
        legend.append({
            "id": cid,
            "label": f"Cluster {cid}" if cid >= 0 else "Noise",
            "color": _cluster_color(cid),
            "count": counts[cid],
        })
    return legend


# ─── Step 4b: Build Radial Tree Data ─────────────────────────────────────────

def build_radial_data(data: dict) -> str:
    """
    From the HDBSCAN condensed tree produce a compact JSON blob:
      { rows: [{p,c,l,s}, ...],   // parent, child, lambda, child_size
        nodeColors: {nodeId: hexColor, ...} }  // dominant-cluster colour per internal node
    Returns the JSON string (or a minimal fallback if no tree available).
    """
    from collections import defaultdict, Counter

    df_ct = data.get("condensed_tree")
    labels = data["labels"]
    n_books = len(labels)

    if df_ct is None or df_ct.empty:
        return json.dumps({"rows": [], "nodeColors": {}}, separators=(",", ":"))

    rows = [
        {"p": int(r.parent), "c": int(r.child),
         "l": round(float(r.lambda_val), 6), "s": int(r.child_size)}
        for r in df_ct.itertuples(index=False)
    ]

    # Build parent→children map to compute dominant cluster for internal nodes
    children_map: dict[int, list[int]] = defaultdict(list)
    for r in rows:
        children_map[r["p"]].append(r["c"])

    # DFS: collect leaf book indices under each internal node
    def get_leaves(node: int) -> list[int]:
        if node < n_books:
            return [node]
        result: list[int] = []
        for ch in children_map.get(node, []):
            result.extend(get_leaves(ch))
        return result

    internal_nodes = {r["p"] for r in rows}  # all parent IDs = internal nodes
    node_colors: dict[str, str] = {}
    for node in internal_nodes:
        leaf_clusters = [int(labels[l]) for l in get_leaves(node) if int(labels[l]) >= 0]
        if leaf_clusters:
            dominant = Counter(leaf_clusters).most_common(1)[0][0]
            node_colors[str(node)] = _cluster_color(dominant)
        else:
            node_colors[str(node)] = "#445566"

    blob = {"rows": rows, "nodeColors": node_colors}
    return json.dumps(blob, separators=(",", ":"))


# ─── Step 5: Generate HTML ────────────────────────────────────────────────────

def generate_html(
    books_json: list,
    ct_json: dict,
    cluster_legend: list,
    cluster_labels: dict,
    model_family: str,
    model_size: str,
    top_k: int,
    radial_blob: str = "{\"rows\":[],\"nodeColors\":{}}",
) -> str:
    data_blob = json.dumps({
        "books": books_json,
        "cluster_topics": ct_json,
        "top_k": top_k,
    }, separators=(",", ":"))

    legend_blob = json.dumps(cluster_legend, separators=(",", ":"))
    labels_blob = json.dumps(cluster_labels, separators=(",", ":"))

    # radial_blob already a JSON string — embed verbatim
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Book Explorer — {model_family} {model_size}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: "Segoe UI", Arial, sans-serif; background: #0f1117; color: #e0e0e0; }}
  header {{ padding: 18px 28px; background: #1a1d27; border-bottom: 1px solid #2e3250; }}
  header h1 {{ font-size: 1.3rem; font-weight: 600; color: #c9d1ff; }}
  header span {{ font-size: 0.85rem; color: #7b87c4; margin-left: 12px; }}
  .controls {{ display: flex; align-items: center; gap: 14px; padding: 14px 28px;
               background: #14172b; border-bottom: 1px solid #2e3250; flex-wrap: wrap; }}
  .controls label {{ font-size: 0.85rem; color: #9aa4d4; }}
  #book-search {{ background: #1e2238; color: #e0e0e0; border: 1px solid #3d4470;
                  border-radius: 6px; padding: 7px 12px; font-size: 0.9rem;
                  width: 260px; }}
  #book-search:focus {{ outline: none; border-color: #6272c4; }}
  #book-search::placeholder {{ color: #555e8a; }}
  #search-count {{ font-size: 0.78rem; color: #555e8a; white-space: nowrap; }}
  #book-select {{ background: #1e2238; color: #e0e0e0; border: 1px solid #3d4470;
                  border-radius: 6px; padding: 7px 12px; font-size: 0.9rem;
                  min-width: 320px; max-width: 520px; cursor: pointer; }}
  #book-select:focus {{ outline: none; border-color: #6272c4; }}
  .info-bar {{ padding: 10px 28px; background: #12152a; font-size: 0.82rem;
               color: #7b87c4; min-height: 36px; }}
  .plots {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px;
            padding: 10px 18px; height: 45vh; }}
  .plot-box {{ background: #1a1d27; border-radius: 10px; border: 1px solid #2e3250;
               overflow: hidden; }}
  .plot-title {{ font-size: 0.78rem; font-weight: 600; color: #7b87c4;
                 letter-spacing: 0.06em; text-transform: uppercase;
                 padding: 8px 14px 0; }}
  .plot-inner {{ width: 100%; height: calc(100% - 28px); }}
  .legend {{ padding: 10px 28px 14px; display: flex; flex-wrap: wrap; gap: 8px; }}
  .legend-item {{ display: flex; align-items: center; gap: 5px; font-size: 0.75rem;
                  color: #9aa4d4; cursor: default; }}
  .legend-dot {{ width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }}
  .radial-section {{ padding: 0 18px 18px; height: 90vh; }}
  .radial-section .plot-box {{ height: 100%; }}
  @media (max-width: 900px) {{ .plots {{ grid-template-columns: 1fr; height: auto; }} }}
</style>
</head>
<body>

<header>
  <h1>📚 Book Explorer
    <span>{model_family} · {model_size} · top-{top_k} neighbours</span>
  </h1>
</header>

<div class="controls">
  <label for="book-search">Search:</label>
  <input id="book-search" type="text" placeholder="Type title or cluster…" autocomplete="off"/>
  <span id="search-count"></span>
  <label for="book-select">Select:</label>
  <select id="book-select"></select>
</div>

<div class="info-bar" id="info-bar">← Select a book to explore its neighbourhood</div>

<div class="plots">
  <div class="plot-box">
    <div class="plot-title">Nearest Neighbors in Embedding Space</div>
    <div class="plot-inner" id="ego-plot"></div>
  </div>
  <div class="plot-box">
    <div class="plot-title">UMAP Space — global position</div>
    <div class="plot-inner" id="umap-plot"></div>
  </div>
</div>

<div class="legend" id="legend"></div>

<div class="radial-section">
  <div class="plot-box">
    <div class="plot-title">Radial Ancestry Tree — condensed HDBSCAN hierarchy (inner = more similar)</div>
    <div class="plot-inner" id="radial-plot"></div>
  </div>
</div>

<script>
// ── Embedded Data ────────────────────────────────────────────────────────────
const DATA = {data_blob};
const LEGEND = {legend_blob};
const CTREE = {radial_blob};
const CLUSTER_LABELS = {labels_blob};
const BOOKS = DATA.books;
const CT = DATA.cluster_topics;
const TOP_K = DATA.top_k;
const N_BOOKS = BOOKS.length;

// ── Build Dropdown (sorted: cluster asc, then title alpha) ──────────────────
const sel = document.getElementById("book-select");
const searchBox = document.getElementById("book-search");
const searchCount = document.getElementById("search-count");

const sortedIndices = BOOKS.map((b, i) => i).sort((a, b) => {{
  const ca = BOOKS[a].cluster, cb = BOOKS[b].cluster;
  const ka = ca < 0 ? Infinity : ca;
  const kb = cb < 0 ? Infinity : cb;
  if (ka !== kb) return ka - kb;
  return BOOKS[a].title.localeCompare(BOOKS[b].title);
}});

// Store all options so we can re-filter without rebuilding from scratch
const allOptions = sortedIndices.map(i => {{
  const b = BOOKS[i];
  const clusterLabel = b.cluster >= 0 ? `C${{b.cluster}}` : "noise";
  const shortTitle = b.title.split(/ by /i)[0];
  const opt = document.createElement("option");
  opt.value = i;
  opt.textContent = `[${{clusterLabel}}] ${{shortTitle.length > 80 ? shortTitle.slice(0,79)+"\u2026" : shortTitle}}`;
  opt._search = (b.title + " " + clusterLabel).toLowerCase();
  return opt;
}});
allOptions.forEach(o => sel.appendChild(o));
searchCount.textContent = `${{allOptions.length}} books`;

// Live search filter
searchBox.addEventListener("input", () => {{
  const q = searchBox.value.trim().toLowerCase();
  sel.innerHTML = "";
  const matched = q ? allOptions.filter(o => o._search.includes(q)) : allOptions;
  matched.forEach(o => sel.appendChild(o));
  searchCount.textContent = `${{matched.length}} / ${{allOptions.length}}`;
  if (matched.length > 0) {{
    sel.value = matched[0].value;
    onBookChange();
  }}
}});

// ── Build Legend ─────────────────────────────────────────────────────────────
const legendEl = document.getElementById("legend");
LEGEND.forEach(item => {{
  const div = document.createElement("div");
  div.className = "legend-item";
  div.title = `${{item.label}} (${{item.count}} books)`;
  div.innerHTML = `<div class="legend-dot" style="background:${{item.color}}"></div>${{item.label}} (${{item.count}})`;
  legendEl.appendChild(div);
}});

// ── Layout helpers ───────────────────────────────────────────────────────────
const DARK_LAYOUT = {{
  paper_bgcolor: "#1a1d27",
  plot_bgcolor: "#1a1d27",
  font: {{ color: "#9aa4d4", size: 11 }},
  margin: {{ l: 30, r: 20, t: 20, b: 30 }},
  showlegend: false,
  xaxis: {{ showgrid: false, zeroline: false, showticklabels: false }},
  yaxis: {{ showgrid: false, zeroline: false, showticklabels: false }},
}};

function clamp(v, lo, hi) {{ return Math.max(lo, Math.min(hi, v)); }}

// ── Ego-Network Renderer ─────────────────────────────────────────────────────
function renderEgo(bookIdx) {{
  const book = BOOKS[bookIdx];
  const nbrs = book.neighbors;
  const n = nbrs.length;

  // Node positions: center at (0,0), neighbours on unit circle
  const nodeX = [0];
  const nodeY = [0];
  const nodeColors = [book.color];
  const nodeSizes = [14];   // uniform — center node same size as neighbours
  const nodeTexts = [book.title];
  const nodeSymbols = ["star"];
  const nodeBorders = ["#ffffff"];
  const nodeKeywords = [book.keywords.join(", ") || "—"];

  // Normalise similarities so the full radius/width range is always used,
  // regardless of how tightly bunched the raw cosine scores are.
  const sims = nbrs.map(nb => nb.sim);
  const simMin = Math.min(...sims);
  const simMax = Math.max(...sims);
  const simRange = simMax - simMin || 1e-6;
  const normSim = sims.map(s => (s - simMin) / simRange); // 0 = least similar, 1 = most similar

  nbrs.forEach((nb, k) => {{
    const angle = (2 * Math.PI * k) / n;
    // Length encodes similarity: most similar (normSim=1) → shortest arm (r=0.30),
    // least similar (normSim=0) → longest arm (r=1.10).
    const r = 0.30 + 0.80 * (1 - normSim[k]);
    nodeX.push(r * Math.cos(angle));
    nodeY.push(r * Math.sin(angle));
    const nb_book = BOOKS[nb.idx];
    nodeColors.push(nb_book.color);
    nodeSizes.push(14);   // uniform node size
    nodeTexts.push(nb_book.title);
    nodeSymbols.push("circle");
    nodeBorders.push(nb_book.cluster === book.cluster ? "#ffffff" : "#ff9944");
    nodeKeywords.push(nb_book.keywords.join(", ") || "—");
  }});

  // Edge traces: uniform thickness; length alone encodes similarity
  const edgeTraces = nbrs.map((nb, k) => {{
    return {{
      type: "scatter", mode: "lines",
      x: [0, nodeX[k + 1]],
      y: [0, nodeY[k + 1]],
      line: {{ width: 1.5, color: "rgba(140,160,220,0.6)" }},
      hoverinfo: "none",
    }};
  }});

  // (similarity labels removed — length encodes similarity visually)

  // Node hover text: Cluster + Cluster topics only
  const hoverTexts = nodeTexts.map((t, i) => {{
    const clusterKey = i === 0
      ? (book.cluster >= 0 ? `cluster_${{book.cluster}}` : "noise")
      : (BOOKS[nbrs[i-1].idx].cluster >= 0 ? `cluster_${{BOOKS[nbrs[i-1].idx].cluster}}` : "noise");
    const clusterTopics = CT[clusterKey] ? CT[clusterKey].join(", ") : "noise";
    return `<b>Cluster:</b> ${{clusterKey}}<br><b>Cluster topics:</b> ${{clusterTopics}}`;
  }});

  // Strip author ("by …") from labels; place text away from centre to reduce overlap
  function angleToTextPos(angle) {{
    // angle in radians; map to 8 Plotly textposition values
    const deg = ((angle * 180 / Math.PI) % 360 + 360) % 360;
    if (deg < 22.5 || deg >= 337.5)  return "middle right";
    if (deg < 67.5)                   return "bottom right";
    if (deg < 112.5)                  return "bottom center";
    if (deg < 157.5)                  return "bottom left";
    if (deg < 202.5)                  return "middle left";
    if (deg < 247.5)                  return "top left";
    if (deg < 292.5)                  return "top center";
    return "top right";
  }}
  const angles = [Math.PI / 2].concat(  // center gets top-center
    nbrs.map((_, k) => (2 * Math.PI * k) / n)
  );
  const labelTexts = nodeTexts.map(t => t.split(/ by /i)[0]);

  const nodeTrace = {{
    type: "scatter", mode: "markers+text",
    x: nodeX, y: nodeY,
    text: labelTexts,
    textposition: angles.map(angleToTextPos),
    textfont: {{ size: 11, color: "#c9d1ff" }},
    marker: {{
      size: nodeSizes,
      color: nodeColors,
      symbol: nodeSymbols,
      line: {{ color: nodeBorders, width: 1.5 }},
    }},
    hovertemplate: "%{{customdata}}<extra></extra>",
    customdata: hoverTexts,
  }};

  Plotly.react("ego-plot",
    [...edgeTraces, nodeTrace],
    Object.assign({{}}, DARK_LAYOUT, {{
      xaxis: Object.assign({{}}, DARK_LAYOUT.xaxis, {{ range: [-1.6, 1.6] }}),
      yaxis: Object.assign({{}}, DARK_LAYOUT.yaxis, {{ range: [-1.6, 1.6], scaleanchor: "x" }}),
    }}),
    {{ responsive: true, displayModeBar: false }}
  );
}}

// ── UMAP Scatter Renderer ────────────────────────────────────────────────────
// Precompute background traces grouped by cluster (static, rendered once)
let umapBackground = null;

function buildUmapBackground() {{
  // One trace per cluster for consistent coloring
  const byCluster = {{}};
  BOOKS.forEach(b => {{
    const k = b.cluster;
    if (!byCluster[k]) byCluster[k] = {{ x:[], y:[], text:[], ids:[] }};
    byCluster[k].x.push(b.umap_x);
    byCluster[k].y.push(b.umap_y);
    byCluster[k].text.push(b.title);
    byCluster[k].ids.push(b.id);
  }});

  return Object.entries(byCluster).map(([cid, pts]) => ({{
    type: "scatter", mode: "markers",
    x: pts.x, y: pts.y,
    name: cid >= 0 ? `Cluster ${{cid}}` : "Noise",
    marker: {{
      size: 5,
      color: parseInt(cid) >= 0 ? BOOKS.find(b => b.cluster === parseInt(cid)).color : "#555",
      opacity: 0.4,
      line: {{ width: 0 }},
    }},
    hovertemplate: "%{{customdata}}<extra></extra>",
    customdata: pts.ids.map(id => {{
      const b = BOOKS[id];
      const kws = b.keywords.slice(0,5).join(", ") || "—";
      return `${{b.title.split(/ by /i)[0]}}<br>Cluster: ${{b.cluster}}<br>Top topics: ${{kws}}`;
    }}),
    ids: pts.ids.map(String),
  }}));
}}

function renderUmap(bookIdx) {{
  if (!umapBackground) umapBackground = buildUmapBackground();

  const book = BOOKS[bookIdx];
  const nbrIds = new Set(book.neighbors.map(n => n.idx));

  // Highlight trace: neighbours
  const nbrBooks = book.neighbors.map(n => BOOKS[n.idx]);
  const nbrTrace = {{
    type: "scatter", mode: "markers",
    x: nbrBooks.map(b => b.umap_x),
    y: nbrBooks.map(b => b.umap_y),
    name: "Neighbours",
    marker: {{
      size: 10,
      color: nbrBooks.map(b => b.color),
      opacity: 1.0,
      line: {{ color: "#ff9944", width: 1.5 }},
      symbol: "circle",
    }},
    hovertemplate: "%{{customdata}}<extra></extra>",
    customdata: nbrBooks.map(b => {{
      const kws = b.keywords.slice(0,5).join(", ") || "—";
      return `${{b.title.split(/ by /i)[0]}}<br>Cluster: ${{b.cluster}}<br>Top topics: ${{kws}}`;
    }}),
  }};

  // Selected book trace
  const selTrace = {{
    type: "scatter", mode: "markers+text",
    x: [book.umap_x], y: [book.umap_y],
    name: "Selected",
    text: [book.title.split(" by ")[0].slice(0,25)],
    textposition: "top center",
    textfont: {{ size: 10, color: "#ffffff" }},
    marker: {{
      size: 16, color: book.color,
      symbol: "star",
      line: {{ color: "#ffffff", width: 2 }},
      opacity: 1.0,
    }},
    hovertemplate: `${{book.title.split(/ by /i)[0]}}<br>Cluster: ${{book.cluster}}<br>Top topics: ${{book.keywords.slice(0,5).join(", ")}}<extra></extra>`,
  }};

  Plotly.react("umap-plot",
    [...umapBackground, nbrTrace, selTrace],
    Object.assign({{}}, DARK_LAYOUT),
    {{ responsive: true, displayModeBar: false }}
  );
}}

// ── Radial Ancestry Tree Renderer (circular dendrogram) ─────────────────────
function renderRadial(bookIdx) {{
  if (!CTREE.rows || CTREE.rows.length === 0) return;

  // ── Build maps ──
  const childToParent = {{}}, parentToChildren = {{}};
  CTREE.rows.forEach(r => {{
    childToParent[r.c] = {{parent: r.p, lambda: r.l}};
    if (!parentToChildren[r.p]) parentToChildren[r.p] = [];
    parentToChildren[r.p].push({{child: r.c, lambda: r.l}});
  }});

  // ── Walk up from selected book to root ──
  const spine = [];
  let curr = bookIdx;
  while (childToParent[curr]) {{
    const {{parent, lambda}} = childToParent[curr];
    spine.push({{nodeId: parent, lambda, fromChild: curr}});
    curr = parent;
  }}
  if (spine.length === 0) {{
    Plotly.react("radial-plot", [], DARK_LAYOUT, {{responsive:true, displayModeBar:false}});
    return;
  }}

  // ── Build subtree with explicit tree depth ──
  function buildSubtree(nodeId, depth) {{
    if (nodeId < N_BOOKS) return {{id: nodeId, isLeaf: true, depth, children: []}};
    const kids = (parentToChildren[nodeId] || []).map(c => buildSubtree(c.child, depth + 1));
    return {{id: nodeId, isLeaf: false, depth, children: kids}};
  }}

  // Branch points: spine nodes that have sibling subtrees
  const branchPoints = spine
    .map(({{nodeId, lambda, fromChild}}, k) => {{
      const sibs = (parentToChildren[nodeId] || []).filter(c => c.child !== fromChild);
      return {{nodeId, depth: k + 1, subtrees: sibs.map(s => buildSubtree(s.child, k + 2))}};
    }})
    .filter(b => b.subtrees.length > 0);

  if (branchPoints.length === 0) {{
    Plotly.react("radial-plot", [], DARK_LAYOUT, {{responsive:true, displayModeBar:false}});
    return;
  }}

  // ── Depth → radius ──
  const allDepths = [];
  function gatherDepths(n) {{ allDepths.push(n.depth); if (!n.isLeaf) n.children.forEach(gatherDepths); }}
  branchPoints.forEach(b => {{ allDepths.push(b.depth); b.subtrees.forEach(gatherDepths); }});
  const maxDepth = Math.max(...allDepths);
  const R_MAX = 1.0;
  function depthR(d) {{ return (d / maxDepth) * R_MAX; }}

  // ── Leaf order: DFS across all branch subtrees ──
  const leafOrder = [];
  function collectLeaves(n) {{ if (n.isLeaf) leafOrder.push(n); else n.children.forEach(collectLeaves); }}
  branchPoints.forEach(b => b.subtrees.forEach(collectLeaves));
  if (leafOrder.length === 0) return;
  const arcPerLeaf = 2 * Math.PI / leafOrder.length;
  const leafAngle = {{}};
  leafOrder.forEach((leaf, i) => {{ leafAngle[leaf.id] = -Math.PI / 2 + i * arcPerLeaf; }});

  // ── Cluster arc centroids for labels ─────────────────────────────────────
  const clusterArcMap = {{}};
  leafOrder.forEach(leaf => {{
    const b = BOOKS[leaf.id];
    const ck = b.cluster >= 0 ? `cluster_${{b.cluster}}` : "noise";
    if (!clusterArcMap[ck]) clusterArcMap[ck] = {{ angles: [], color: b.color, maxDepth: 0 }};
    clusterArcMap[ck].angles.push(leafAngle[leaf.id]);
    if (leaf.depth > clusterArcMap[ck].maxDepth) clusterArcMap[ck].maxDepth = leaf.depth;
  }});
  const labelItems = [];
  Object.entries(clusterArcMap).forEach(([ck, {{angles, color, maxDepth}}]) => {{
    const lbl = CLUSTER_LABELS[ck];
    if (!lbl) return;
    const midA = (Math.min(...angles) + Math.max(...angles)) / 2;
    const rTip  = depthR(maxDepth);          // arrowhead at outermost leaf
    const rText = depthR(maxDepth) + 0.16;   // text a little further out
    labelItems.push({{ ck, midA, rTip, rText, label: lbl, color }});
  }});
  // Word-wrap: prefer wide lines (~38 chars) so text stays compact vertically
  function wrapLabel(text, maxLen) {{
    if (text.length <= maxLen) return text;
    const words = text.split(" ");
    const lines = []; let line = "";
    words.forEach(w => {{
      if (line.length + w.length + 1 > maxLen && line.length > 0) {{
        lines.push(line); line = w;
      }} else {{
        line = line ? line + " " + w : w;
      }}
    }});
    if (line) lines.push(line);
    return lines.join("<br>");
  }}
  const annotations = labelItems.map(item => {{
    const cosA = Math.cos(item.midA), sinA = Math.sin(item.midA);
    const xa = cosA > 0.12 ? "left" : cosA < -0.12 ? "right" : "center";
    const ya = sinA > 0.15 ? "bottom" : sinA < -0.15 ? "top" : "middle";
    return {{
      // arrow tip → cluster position
      x: item.rTip * cosA,
      y: item.rTip * sinA,
      // text anchor, in data coords
      ax: item.rText * cosA,
      ay: item.rText * sinA,
      axref: "x", ayref: "y",
      text: wrapLabel(item.label, 38),
      showarrow: true,
      arrowhead: 2,
      arrowsize: 0.8,
      arrowwidth: 1,
      arrowcolor: item.color,
      font: {{ size: 9.5, color: item.color }},
      bgcolor: "rgba(15,17,23,0.82)",
      borderpad: 2,
      xanchor: xa,
      yanchor: ya,
      align: "center",
    }};
  }});

  // ── Angles for internal nodes (midpoint of children's angle span) ──
  const nodeAngle = {{}};
  function computeAngles(node) {{
    if (node.isLeaf) {{ nodeAngle[node.id] = leafAngle[node.id]; return; }}
    node.children.forEach(computeAngles);
    const cas = node.children.map(c => nodeAngle[c.id]);
    nodeAngle[node.id] = (Math.min(...cas) + Math.max(...cas)) / 2;
  }}
  branchPoints.forEach(b => {{
    b.subtrees.forEach(computeAngles);
    const sas = b.subtrees.map(s => nodeAngle[s.id]);
    nodeAngle[b.nodeId] = (Math.min(...sas) + Math.max(...sas)) / 2;
  }});

  // ── Drawing helpers ──
  const ex = [], ey = [];
  function addArc(r, a0, a1) {{
    if (Math.abs(a1 - a0) < 1e-9) return;
    const steps = Math.max(2, Math.ceil(Math.abs(a1 - a0) / (Math.PI / 60)));
    for (let i = 0; i <= steps; i++) {{
      const a = a0 + (a1 - a0) * i / steps;
      ex.push(r * Math.cos(a)); ey.push(r * Math.sin(a));
    }}
    ex.push(null); ey.push(null);
  }}
  function addSpoke(r1, r2, a) {{
    ex.push(r1 * Math.cos(a), r2 * Math.cos(a), null);
    ey.push(r1 * Math.sin(a), r2 * Math.sin(a), null);
  }}

  // ── L-shaped connectors for an internal node ──
  function drawNode(node) {{
    if (node.isLeaf) return;
    const r = depthR(node.depth);
    const cas = node.children.map(c => nodeAngle[c.id]);
    addArc(r, Math.min(...cas), Math.max(...cas));
    node.children.forEach(child => {{
      addSpoke(r, depthR(child.depth), nodeAngle[child.id]);
      drawNode(child);
    }});
  }}

  // ── Spine + sibling branches ──
  branchPoints.forEach((bp, bi) => {{
    const bR = depthR(bp.depth), bA = nodeAngle[bp.nodeId];
    const prevR = bi === 0 ? 0 : depthR(branchPoints[bi - 1].depth);
    const prevA = bi === 0 ? bA : nodeAngle[branchPoints[bi - 1].nodeId];
    if (bi > 0) addArc(prevR, Math.min(prevA, bA), Math.max(prevA, bA));
    addSpoke(prevR, bR, bA);
    const sas = bp.subtrees.map(s => nodeAngle[s.id]);
    addArc(bR, Math.min(Math.min(...sas), bA), Math.max(Math.max(...sas), bA));
    bp.subtrees.forEach(sub => {{
      addSpoke(bR, depthR(sub.depth), nodeAngle[sub.id]);
      drawNode(sub);
    }});
  }});

  // ── Leaf book nodes ──
  const leafX = [], leafY = [], leafColors = [], leafHovers = [];
  leafOrder.forEach(leaf => {{
    const b = BOOKS[leaf.id], a = leafAngle[leaf.id], r = depthR(leaf.depth);
    leafX.push(r * Math.cos(a)); leafY.push(r * Math.sin(a));
    leafColors.push(b.color);
    const ck = b.cluster >= 0 ? `cluster_${{b.cluster}}` : "noise";
    leafHovers.push(`${{b.title.split(/ by /i)[0]}}<br>Cluster: ${{b.cluster}}<br>Topics: ${{CT[ck] ? CT[ck].join(", ") : "—"}}`);
  }});

  // ── Internal node dots ──
  const intX = [], intY = [], intC = [];
  function collectInt(node) {{
    if (node.isLeaf) return;
    const r = depthR(node.depth), a = nodeAngle[node.id];
    intX.push(r * Math.cos(a)); intY.push(r * Math.sin(a));
    intC.push(CTREE.nodeColors[String(node.id)] || "#445566");
    node.children.forEach(collectInt);
  }}
  branchPoints.forEach(b => {{
    intX.push(depthR(b.depth) * Math.cos(nodeAngle[b.nodeId]));
    intY.push(depthR(b.depth) * Math.sin(nodeAngle[b.nodeId]));
    intC.push(CTREE.nodeColors[String(b.nodeId)] || "#445566");
    b.subtrees.forEach(collectInt);
  }});

  // ── Selected book (centre star) ──
  const cb = BOOKS[bookIdx];
  const cbK = cb.cluster >= 0 ? `cluster_${{cb.cluster}}` : "noise";

  Plotly.react("radial-plot",
    [
      {{type:"scatter", mode:"lines",
        x:ex, y:ey, line:{{width:0.9, color:"rgba(110,130,210,0.55)"}}, hoverinfo:"none"}},
      {{type:"scatter", mode:"markers",
        x:leafX, y:leafY,
        marker:{{size:8, color:leafColors, opacity:0.9, line:{{width:0}}}},
        hovertemplate:"%{{customdata}}<extra></extra>", customdata:leafHovers}},
      {{type:"scatter", mode:"markers",
        x:intX, y:intY,
        marker:{{size:4, color:intC, opacity:0.7, symbol:"circle-open",
                 line:{{width:1.5, color:intC}}}},
        hoverinfo:"none"}},
      {{type:"scatter", mode:"markers",
        x:[0], y:[0],
        marker:{{size:18, color:cb.color, symbol:"star",
                 line:{{color:"#ffffff",width:2}}, opacity:1}},
        hovertemplate:`${{cb.title.split(/ by /i)[0]}}<br>Cluster: ${{cb.cluster}}<br>Topics: ${{CT[cbK] ? CT[cbK].join(", ") : "—"}}<extra></extra>`}},
    ],
    Object.assign({{}}, DARK_LAYOUT, {{
      xaxis: Object.assign({{}}, DARK_LAYOUT.xaxis, {{range:[-1.35, 1.35]}}),
      yaxis: Object.assign({{}}, DARK_LAYOUT.yaxis, {{range:[-1.35, 1.35], scaleanchor:"x"}}),
      dragmode: "pan",
      annotations,
    }}),
    {{responsive:true, displayModeBar:"hover", scrollZoom:true}}
  );
}}

// ── Info Bar ─────────────────────────────────────────────────────────────────
function updateInfoBar(bookIdx) {{
  const book = BOOKS[bookIdx];
  const sameClusterNbrs = book.neighbors.filter(n => BOOKS[n.idx].cluster === book.cluster).length;
  const clusterKey = book.cluster >= 0 ? `cluster_${{book.cluster}}` : "noise";
  const clusterTopics = CT[clusterKey] ? CT[clusterKey].join(" · ") : "noise";
  document.getElementById("info-bar").innerHTML =
    `<b>${{book.title}}</b> &nbsp;·&nbsp; Genre: ${{book.genre}} &nbsp;·&nbsp; ` +
    `Cluster: <span style="color:${{book.color}}">${{book.cluster >= 0 ? book.cluster : "noise"}}</span> ` +
    `(${{sameClusterNbrs}}/${{book.neighbors.length}} neighbours in same cluster) &nbsp;·&nbsp; ` +
    `Cluster topics: <i>${{clusterTopics}}</i>`;
}}

// ── Main Update ───────────────────────────────────────────────────────────────
function onBookChange() {{
  const idx = parseInt(sel.value);
  renderEgo(idx);
  renderUmap(idx);
  renderRadial(idx);
  updateInfoBar(idx);
}}

sel.addEventListener("change", onBookChange);

// Init with first book
onBookChange();
</script>
</body>
</html>"""
    return html


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Emergence Book Visualizer")
    parser.add_argument("--model_family", type=str, default="Qwen3-Embedding")
    parser.add_argument("--model_size", type=str, default="0.6B")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Number of nearest neighbours to show per book")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Visualization: {args.model_family} / {args.model_size}")
    print(f"{'='*60}\n")

    # Load
    data = load_data(args.model_family, args.model_size)

    # Compute top-K neighbours
    neighbors = compute_neighbors(data["embeddings"], top_k=args.top_k)

    # Build data structures
    books_json, ct_json, cluster_labels = build_book_data(data, neighbors)
    cluster_legend = build_cluster_legend(data["labels"])

    # Build radial tree data
    radial_blob = build_radial_data(data)

    # Generate HTML
    print(f"[html] Generating book explorer HTML ...")
    html = generate_html(
        books_json, ct_json, cluster_legend, cluster_labels,
        args.model_family, args.model_size, args.top_k,
        radial_blob=radial_blob,
    )

    # Save
    out_dir = os.path.join(data["run_dir"], "per-book-relation")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "book_explorer.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    size_kb = os.path.getsize(out_path) / 1024
    print(f"[html] Saved to {out_path}  ({size_kb:.0f} KB)")
    print(f"\n[done] Open in a browser or VS Code Simple Browser.")


if __name__ == "__main__":
    main()
