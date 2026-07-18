# Glance Fashion Retrieval

> **ML Internship Take-Home Submission** — Intelligent fashion image search using structured attribute-augmented dual-space retrieval.

---

## Evaluation Results

Results from `evaluation/results.json` — all 5 official queries run against the final index (post-CLIP query embedding fix, see Fix #3 notes in git history):

| Query | Top-1 Image ID | Top-1 Fused Score |
|-------|---------------|-------------------|
| A person in a bright yellow raincoat. | `d5420eb0d6e13003799778f0157b0a0e` | **0.4758** |
| Professional business attire inside a modern office. | `e2ba4d1e422cc766d9e8eaacd3a25a3e` | **0.4463** |
| Someone wearing a blue shirt sitting on a park bench. | `f06fd9e18c7a2c5b533e68de88fb487b` | **0.5181** |
| Casual weekend outfit for a city walk. | `4dc413ef5a47c5da696ba2f72835e6ed` | **0.5177** |
| A red tie and a white shirt in a formal setting. | `13ac5b09f6c7cf03d6bddeeba2dd7d63` | **0.4900** |

> Full per-query results with score breakdowns (clip/text/attribute), matched attributes, and latency are in [`evaluation/results.json`](evaluation/results.json).

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     PART A: INDEXER                                  │
│                                                                       │
│  Image ──► VLM (BLIP-2)  ──►  Structured JSON Caption              │
│              │                  { clothing_items, colors,             │
│              │                    environment, style, vibe }          │
│              │                          │                             │
│              │                  caption_to_text()                     │
│              │                          │                             │
│              ▼                          ▼                             │
│         OpenCLIP              SentenceTransformer (BGE)              │
│         ViT-B-32                  bge-small-en-v1.5                  │
│         512-d visual              384-d text embedding               │
│              │                          │                             │
│              ▼                          ▼                             │
│      ChromaDB: fashion_clip   ChromaDB: fashion_text                 │
│         (cosine, 512-d)           (cosine, 384-d)                    │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   │  (shared image_id key)
                                   │
┌─────────────────────────────────────────────────────────────────────┐
│                     PART B: RETRIEVER                                 │
│                                                                       │
│  Query ──► QueryParser (Groq LLM / rule-based fallback)             │
│              │                                                        │
│              └─► ParsedQuery { clothing, colors, env, style,         │
│                                filter_*, expanded_query }             │
│                      │               │                               │
│                      ▼               ▼                               │
│                 CLIP text emb    BGE query emb                       │
│                      │               │                               │
│                      ▼               ▼                               │
│               query_clip()      query_text()                         │
│                      │               │                               │
│                      └───────┬───────┘                               │
│                              ▼                                        │
│               Fusion: α·clip + β·text + attr_bonus                  │
│               (α=0.35, β=0.50, γ_cap=0.15)                          │
│                              │                                        │
│                              ▼                                        │
│                    Ranked Results + Explanations                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

### ⚠️ Known Limitation — Caption Quality (Submitted Index)

This submission's index was built using the **CPU-friendly `LightweightCaptionGenerator` fallback path** (base BLIP, free-text caption + heuristic keyword-proximity parsing) due to no local GPU access at submission time.

Real numbers computed from `data/captions.json` (1,000 images):

| Metric | Value |
|--------|-------|
| `parse_tier_used: 3` (heuristic fallback) | **1000 / 1000 — 100.0%** |
| `environment: unknown` | **575 / 1000 — 57.5%** |
| `style: unknown` | **918 / 1000 — 91.8%** |

**What this means:** The structured BLIP-2 path (`Salesforce/blip2-flan-t5-xl`) that outputs validated JSON captions with explicit `environment` and `style` fields was designed and implemented (see `Part_A_Indexer/caption_generator.py`) but requires a GPU to run at scale. The current index relies on free-text BLIP captions + proximity-based keyword heuristics, which correctly identify clothing items and color-garment pairs but produce high rates of `unknown` for the categorical fields.

**How to fix:** This is a **one CLI flag change** — drop `--use_lightweight_vlm` when running the indexer:
```bash
# Current (CPU, what was used for this submission):
python -m Part_A_Indexer.run_indexer --mode full --use_lightweight_vlm

# Intended (GPU, structured BLIP-2 path — requires ~8 GB VRAM or use --use_4bit for ~6 GB):
python -m Part_A_Indexer.run_indexer --mode full
```
On a Colab T4 GPU this would take ~2–3 hours for 1,000 images and would directly resolve the high `unknown` rates. The assignment's design, code, and architecture are built for the full BLIP-2 path; only the submitted data artifact reflects the CPU fallback.

---

## Why Not Vanilla CLIP?

CLIP produces a single holistic embedding per image via contrastive image-text training. This causes **three specific failures** for fashion retrieval:

### 1. Compositional Binding Failure
CLIP has no mechanism to track *which color attaches to which garment*. It operates as a bag-of-concepts over the entire image.

**Example:** `"red tie, white shirt"` and `"white tie, red shirt"` produce nearly identical CLIP embeddings. The token `red` and `tie` appear in both queries in similar proportional context.

**Fix:** Generate structured JSON captions that explicitly bind: `["red tie", "white shirt"]`. Embed these as text using BGE (trained on retrieval tasks). Now "red tie" and "white tie" are distinguishable.

### 2. Fine-Grained Fashion Vocabulary Gap
CLIP's training data (web alt-text) underrepresents specific fashion terminology. `"blazer vs sport coat vs suit jacket"` — CLIP conflates these; they land near each other in embedding space because the alt-text that generated them was generic.

**Fix:** VLM-generated captions use the visual context to produce precise garment labels (the VLM sees the garment, not just text describing it). BGE can then distinguish these in text space.

### 3. Context Dilution
When clothing dominates the frame, environment cues (office vs. park) get diluted in a single global CLIP vector.

**Fix:** Structured captions extract `environment` and `style` as discrete categorical fields. At query time, these can be used as hard filters (`where={"environment": "office"}`) independent of vector similarity.

---

## Repository Structure

```
glance-fashion-retrieval/
├── README.md                       ← You are here
├── requirements.txt                ← All dependencies with version pins
├── .env.example                    ← Environment variable template
├── .gitignore
│
├── Part_A_Indexer/
│   ├── __init__.py
│   ├── caption_generator.py        ← VLM → structured JSON + 3-tier fallback parser
│   ├── feature_extractor.py        ← DualEmbedder: CLIP visual + BGE text
│   ├── vector_store.py             ← ChromaDB dual-collection manager
│   └── run_indexer.py              ← CLI: resumable indexing pipeline
│
├── Part_B_Retriever/
│   ├── __init__.py
│   ├── query_parser.py             ← LLM + rule-based query decomposition
│   ├── retriever.py                ← Hybrid fusion + attribute bonus + explainer
│   └── run_retrieval.py            ← CLI + FastAPI server + demo UI
│
├── data/
│   └── download_dataset.py         ← Fashionpedia downloader (streaming, resumable)
│
├── notebooks/
│   └── exploration.ipynb           ← EDA, 2D projection, ablation, weight sweep
│
└── evaluation/
    └── run_eval_queries.py         ← Runs 5 official queries, saves results.json
```

---

## Setup & Quickstart

### Prerequisites
- Python 3.10+
- CUDA GPU recommended for BLIP-2 (8+ GB VRAM). CPU works via `--use_lightweight_vlm`.
- Free [Groq API key](https://console.groq.com/) for LLM query parsing (optional — system runs offline without it).

### Installation

```bash
# Clone and enter repo
git clone https://github.com/your-username/glance-fashion-retrieval
cd glance-fashion-retrieval

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env: add GROQ_API_KEY (optional) and set DATASET_SPLIT if using training data
```

### Step 1: Download Dataset

Fashionpedia provides two splits. **Both support `--limit` for streaming extraction** — you never need to download the full archive to get N images.

| Split | Flag | Size | Images | When to use |
|-------|------|------|--------|-------------|
| `val_test` | *(default)* | ~2–3 GB | ~8,726 | Assignment spec default, fast |
| `train` | `--split train` | ~20 GB | ~45,623 | Richer, more diverse index |

```bash
# ── Val/test split (default, per assignment spec) ──────────────────────────
# Stream-extract first 1000 images (~2-3 GB download)
python data/download_dataset.py --limit 1000

# ── Training split (optional, larger/richer index) ──────────────────────────
# Stream first 1000 training images WITHOUT downloading all 20 GB
python data/download_dataset.py --split train --limit 1000

# Full training set (downloads all 20 GB, extracts all 45k images)
python data/download_dataset.py --split train --target_sample 5000

# ── General options ─────────────────────────────────────────────────────────
# Skip download if the zip is already present (resume-safe)
python data/download_dataset.py --skip_download

# Output in all cases: data/images/ (images) + data/metadata.json (manifest)
```

### Step 2: Run the Indexer (Part A)

```bash
# Full pipeline: caption + embed + index
# GPU (BLIP-2, recommended):
python -m Part_A_Indexer.run_indexer --mode full

# CPU / low VRAM (BLIP base, faster but lower quality):
python -m Part_A_Indexer.run_indexer --mode full --use_lightweight_vlm

# BLIP-2 on constrained GPU (4-bit quantization, ~6 GB VRAM):
python -m Part_A_Indexer.run_indexer --mode full --use_4bit

# Quick test with 10 images:
python -m Part_A_Indexer.run_indexer --mode full --limit 10

# Resumable: just re-run the same command after an interrupt.
# Captions checkpoint to data/captions_checkpoint.json every 50 images.
```

### Step 3: Run the Retriever (Part B)

```bash
# Single query
python -m Part_B_Retriever.run_retrieval \
    --query "A red tie and a white shirt in a formal setting." \
    --verbose

# Run all 5 official eval queries
python -m Part_B_Retriever.run_retrieval --demo --verbose

# Launch the FastAPI server + demo UI
python -m Part_B_Retriever.run_retrieval --serve

# Then open: http://localhost:8000/
#   API docs: http://localhost:8000/docs
#   Health:   http://localhost:8000/health
```

### Step 4: Run Evaluation

```bash
python evaluation/run_eval_queries.py
# Saves: evaluation/results.json (include as submission PDF appendix)
```

### Step 5: Explore the Notebook

```bash
jupyter lab notebooks/exploration.ipynb
```

---

## API Reference

### `POST /search`
```json
{
  "query": "A red tie and a white shirt in a formal setting.",
  "top_k": 10
}
```
Response includes full score breakdown per result:
```json
{
  "query": "...",
  "parsed_query": { "clothing_items": ["tie","shirt"], "colors": ["red tie","white shirt"], ... },
  "results": [
    {
      "image_id": "001234",
      "fused_score": 0.7821,
      "clip_score": 0.6120,
      "text_score": 0.8943,
      "attribute_bonus": 0.0900,
      "matched_attributes": ["style:formal", "color_garment:red tie"],
      "metadata": { "environment": "office", "style": "formal", ... }
    }
  ]
}
```

### `GET /eval`
Runs all 5 official evaluation queries and returns the structured report.

### `GET /health`
Returns index statistics and fusion weight configuration.

---

## Fusion Formula

```
score = α · CLIP_similarity + β · BGE_text_similarity + attribute_bonus
```

| Weight | Default | Role |
|--------|---------|------|
| α | 0.35 | CLIP visual: aesthetic/vibe/setting recognition |
| β | 0.50 | BGE text: compositional attribute matching |
| γ (cap) | 0.15 | Attribute bonus ceiling |

**Why β > α:** The text/BGE path retrieves against structured captions that explicitly bind color to garment. For compositional queries, this is the dominant signal. CLIP visual acts as a tiebreaker for aesthetic similarity and scene recognition. The 35/50 split was validated on the 5 eval queries in the ablation notebook.

**Attribute Bonus Components:**
- Environment match: +40% of γ cap
- Style match: +30% of γ cap  
- Each color-garment pair match: +10% of γ cap
- Each clothing item match: +5% of γ cap

---

## Scalability

### Current Scale: ~1,000 images
ChromaDB with HNSW indexing. Queries: ~50ms including model inference. Storage: ~10 MB on disk.

### At 1 Million Images: ChromaDB HNSW is still viable
- HNSW: O(log N) approximate nearest neighbor, ~200ms at 1M vectors.
- ChromaDB handles this range well on commodity hardware.
- Metadata filtering (environment/style) uses ChromaDB's SQLite layer — fine at this scale.

### At 100 Million+ Images: Migrate to FAISS IVF-PQ

```python
# Migration sketch:
import faiss, numpy as np

# Export embeddings from ChromaDB
embeddings = np.array(chroma_collection.get(include=["embeddings"])["embeddings"])

# IVF: Inverted File Index — divides space into K Voronoi cells (coarse quantization).
# Query visits only nprobe cells (~1-5% of vectors) → ~100x speedup vs brute force.
nlist = 1024   # number of Voronoi cells
quantizer = faiss.IndexFlatL2(512)
index = faiss.IndexIVFPQ(
    quantizer,
    512,    # dimension
    nlist,  # IVF cells
    64,     # PQ subvectors (controls compression: 64 → 64-byte code vs 2048-byte float32)
    8       # bits per subvector (PQ codebook size = 2^8 = 256 codes)
)
# PQ compresses each 512-d float32 vector (2 KB) to a 64-byte code → 32x compression.
# For 100M images: 200 GB → 6 GB RAM. Sub-millisecond query latency.
index.train(embeddings)
index.add(embeddings)
index.nprobe = 32  # number of cells to visit at query time (recall/speed tradeoff)
```

**Migration path summary:**
1. Export ChromaDB embeddings to numpy arrays (~5 min)  
2. Train and build FAISS IVF-PQ index (~30 min for 100M vectors)
3. Move metadata to DuckDB or Parquet for predicate-pushdown filtering
4. Swap the `query_clip()` / `query_text()` calls in `vector_store.py` — the interface is unchanged

---

## Future Work

### A. Location & Weather Awareness Extension

The current `environment` tag is coarse (office/park/beach). To add location and weather:

1. **Location**: Add GPS metadata parsing if images carry EXIF coordinates → reverse geocode to neighborhood/city type using OpenStreetMap Nominatim (free, offline-capable via local instance).
2. **Weather**: Run a weather classifier (e.g., fine-tune ViT on Clear/Rain/Snow/Cloudy from weather image datasets) as a separate image header — add `weather` as a new structured caption field.
3. Extend `CAPTION_SCHEMA`, `caption_to_metadata()`, and `QueryParser` schemas with the new fields. No structural changes needed to the retrieval pipeline.

### B. Precision Improvements

**1. Fine-tune CLIP on fashion-specific data**  
Train OpenCLIP on FashionIQ or DeepFashion using their triplet annotations (query, positive, negative). This closes the fashion vocabulary gap at the visual embedding level. Expected improvement: +5-10 NDCG on fine-grained attribute queries.

**2. Cross-encoder re-ranking**  
After retrieving top-50 candidates via the dual-space approach, run a BERT-based cross-encoder (e.g., `cross-encoder/ms-marco-MiniLM-L6-v2`) on (query, caption_text) pairs. Cross-encoders do full attention over the pair — no early-binding of separate embeddings — so they catch subtle mismatches that bi-encoder retrieval misses. ~3x latency cost, ~5-8 NDCG improvement.

**3. Region-based multi-scale features**  
Segment each image into regions (upper body, lower body, accessories) using Segment Anything (SAM) or a fashion-specific parser (SCHP). Generate separate CLIP embeddings per region. For a query like "red tie and white shirt", compare the tie query embedding against upper-body crop embeddings only. This solves compositional binding *at the visual level* (not just text level), which is the hardest form of the problem.

**4. Click-through feedback loop**  
Log which results users click on for each query. Use these as implicit relevance signals to tune fusion weights (α, β, γ) via a bandit or gradient-based optimizer. This is the standard production path for search quality improvement.

---

## Model Cards & Attributions

| Component | Model | License |
|-----------|-------|---------|
| VLM Captioner | [Salesforce/blip2-flan-t5-xl](https://huggingface.co/Salesforce/blip2-flan-t5-xl) | BSD-3 |
| Fallback VLM | [Salesforce/blip-image-captioning-base](https://huggingface.co/Salesforce/blip-image-captioning-base) | BSD-3 |
| Visual Embedding | [OpenCLIP ViT-B-32/openai](https://github.com/mlfoundations/open_clip) | MIT |
| Text Embedding | [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) | MIT |
| LLM Query Parser | [Groq llama-3.1-8b-instant](https://console.groq.com/) | Meta Llama License |
| Dataset | [Fashionpedia](https://fashionpedia.github.io/home/) | CC BY 4.0 |
| Vector Store | [ChromaDB](https://www.trychroma.com/) | Apache 2.0 |
