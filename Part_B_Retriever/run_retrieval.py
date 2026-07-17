"""
Part_B_Retriever/run_retrieval.py
===================================
CLI entrypoint, FastAPI server, and demo HTML UI for the retrieval system (Part B).

DESIGN:
  Single file serves three modes:
  1. CLI: --query "text" runs a single search and prints results
  2. CLI: --demo runs all 5 official eval queries and prints a summary table
  3. FastAPI: --serve launches a local server with:
       GET  /           → Minimal HTML/JS search demo UI (no frontend build step)
       GET  /health     → Health check (returns index stats)
       POST /search     → Ranked results with full score breakdown (JSON)
       GET  /eval       → Run all 5 official queries, return structured report

  The demo UI is inline HTML/JS — no npm, no webpack, no Vite. A grader can
  launch the server and immediately interact with the system in a browser.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

# Force UTF-8 output on Windows (cp1252 default crashes on emoji in print statements).
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent.resolve()
DEFAULT_CHROMA_DIR = REPO_ROOT / "chroma_db"

# The 5 official evaluation queries from the assignment brief
EVAL_QUERIES = [
    "A person in a bright yellow raincoat.",
    "Professional business attire inside a modern office.",
    "Someone wearing a blue shirt sitting on a park bench.",
    "Casual weekend outfit for a city walk.",
    "A red tie and a white shirt in a formal setting.",
]


# ---------------------------------------------------------------------------
# System initializer — shared between CLI and FastAPI paths
# ---------------------------------------------------------------------------

def build_retrieval_system(
    chroma_dir: Path = DEFAULT_CHROMA_DIR,
    alpha: float = 0.35,
    beta: float = 0.50,
    gamma: float = 0.15,
    use_rrf: bool = False,
) -> "HybridRetriever":
    """
    Initialize and return a HybridRetriever instance.

    Separated into a function so it can be called once at FastAPI startup
    (using lifespan context manager) and reused across requests, avoiding
    expensive model reloads on every call.
    """
    from Part_A_Indexer.feature_extractor import DualEmbedder
    from Part_A_Indexer.vector_store import FashionVectorStore
    from Part_B_Retriever.query_parser import QueryParser
    from Part_B_Retriever.retriever import HybridRetriever

    logger.info("[System] Initializing retrieval system...")
    vector_store = FashionVectorStore(persist_dir=chroma_dir)
    embedder = DualEmbedder()
    query_parser = QueryParser()

    retriever = HybridRetriever(
        vector_store=vector_store,
        embedder=embedder,
        query_parser=query_parser,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        use_rrf=use_rrf,
    )
    stats = vector_store.get_collection_stats()
    logger.info(
        f"[System] Ready. CLIP: {stats['clip_vectors']} vectors, "
        f"Text: {stats['text_vectors']} vectors."
    )
    return retriever


# ---------------------------------------------------------------------------
# CLI: single query
# ---------------------------------------------------------------------------

def run_single_query(
    retriever: "HybridRetriever",
    query: str,
    top_k: int = 10,
    verbose: bool = False,
) -> None:
    from Part_B_Retriever.retriever import QueryExplainer

    print(f"\n>> Query: \"{query}\"")
    print("-" * 60)

    results, parsed = retriever.retrieve(query, top_k=top_k)

    if verbose:
        print("\n[Query Parse]")
        print(QueryExplainer.explain_query_parse(parsed))
        print()

    if not results:
        print("[!] No results found. Make sure the index is built (run run_indexer.py first).")
        return

    print(f"\nTop {len(results)} Results:\n")
    for rank, result in enumerate(results, 1):
        print(f"  Rank {rank}: {result.image_id}  (score: {result.fused_score:.4f})")
        if verbose:
            print(QueryExplainer.explain("", parsed, result))
            print()
        else:
            print(f"    Path: {result.image_path}")
            meta = result.metadata
            print(
                f"    [{meta.get('style','?')} | {meta.get('environment','?')} | "
                f"{meta.get('dominant_color','?')}]"
            )
            print()


# ---------------------------------------------------------------------------
# CLI: demo (all 5 eval queries)
# ---------------------------------------------------------------------------

def run_demo(retriever: "HybridRetriever", top_k: int = 5, verbose: bool = False) -> None:
    from Part_B_Retriever.retriever import QueryExplainer

    print("\n" + "="*70)
    print("  DEMO: Running All 5 Official Evaluation Queries")
    print("="*70)

    for i, query in enumerate(EVAL_QUERIES, 1):
        print(f"\n[Query {i}/5] {query}")
        print("-" * 60)
        results, parsed = retriever.retrieve(query, top_k=top_k)

        if verbose:
            print(QueryExplainer.explain_query_parse(parsed))
            print()

        for rank, result in enumerate(results, 1):
            meta = result.metadata
            print(
                f"  #{rank:2d} {result.image_id:20s}  "
                f"score={result.fused_score:.3f}  "
                f"[{meta.get('style','?')[:10]:10s} | "
                f"{meta.get('environment','?')[:12]:12s} | "
                f"{meta.get('dominant_color','?')[:8]:8s}]"
            )
            if verbose and result.matched_attributes:
                print(f"       Matched: {', '.join(result.matched_attributes)}")

    print("\n" + "="*70)


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

def create_fastapi_app(chroma_dir: Path, alpha: float, beta: float, gamma: float, use_rrf: bool):
    """
    Factory function that creates and configures the FastAPI app.

    We use a lifespan context manager to initialize the retriever once at
    startup (heavy model loads) and store it in app.state for request reuse.
    """
    from contextlib import asynccontextmanager
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse, JSONResponse
    from pydantic import BaseModel

    class SearchRequest(BaseModel):
        query: str
        top_k: int = 10
        use_rrf: Optional[bool] = None   # override instance default if set

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Load models once at startup; clean up on shutdown."""
        logger.info("[FastAPI] Starting up — loading retrieval system...")
        app.state.retriever = build_retrieval_system(
            chroma_dir=chroma_dir,
            alpha=alpha, beta=beta, gamma=gamma, use_rrf=use_rrf,
        )
        yield
        logger.info("[FastAPI] Shutting down.")

    app = FastAPI(
        title="Glance Fashion Retrieval API",
        description="Hybrid CLIP + BGE structured-caption fashion image search.",
        version="1.0.0",
        lifespan=lifespan,
    )

    # --- Health ---

    @app.get("/health")
    async def health():
        """Index statistics and system status."""
        retriever = app.state.retriever
        stats = retriever.vector_store.get_collection_stats()
        return {
            "status": "healthy",
            "clip_vectors": stats["clip_vectors"],
            "text_vectors": stats["text_vectors"],
            "fusion_weights": {
                "alpha": retriever.alpha,
                "beta": retriever.beta,
                "gamma": retriever.gamma,
            },
            "use_rrf": retriever.use_rrf,
        }

    # --- Search ---

    @app.post("/search")
    async def search(request: SearchRequest):
        """
        Run a hybrid fashion image search.

        Returns ranked results with full score breakdown for interpretability.
        """
        from Part_B_Retriever.retriever import QueryExplainer

        retriever = app.state.retriever

        # Allow per-request RRF override (useful for A/B testing)
        if request.use_rrf is not None:
            retriever.use_rrf = request.use_rrf

        results, parsed = retriever.retrieve(request.query, top_k=request.top_k)

        return {
            "query":          request.query,
            "parsed_query":   parsed.to_dict(),
            "total_results":  len(results),
            "results": [r.to_dict() for r in results],
        }

    # --- Eval ---

    @app.get("/eval")
    async def eval_endpoint():
        """Run all 5 official evaluation queries and return structured results."""
        retriever = app.state.retriever
        report = {}
        for query in EVAL_QUERIES:
            results, parsed = retriever.retrieve(query, top_k=10)
            report[query] = {
                "parsed_query": parsed.to_dict(),
                "results":      [r.to_dict() for r in results],
            }
        return report

    # --- Demo UI ---

    @app.get("/", response_class=HTMLResponse)
    async def demo_ui():
        """
        Minimal single-file HTML+JS demo UI.
        No build step, no external JS frameworks, no CDN dependencies.
        Works entirely with the browser's built-in fetch API.
        """
        html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Glance Fashion Retrieval</title>
  <style>
    :root {
      --bg: #0f0f13;
      --surface: #1a1a24;
      --accent: #7c6af7;
      --accent2: #f77c6a;
      --text: #e8e8f0;
      --muted: #888899;
      --border: #2a2a3a;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: var(--bg);
      color: var(--text);
      font-family: 'Inter', system-ui, sans-serif;
      min-height: 100vh;
      padding: 2rem;
    }
    h1 {
      font-size: 2rem;
      font-weight: 700;
      background: linear-gradient(135deg, var(--accent), var(--accent2));
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
      margin-bottom: 0.5rem;
    }
    .subtitle { color: var(--muted); margin-bottom: 2rem; font-size: 0.95rem; }
    .search-bar {
      display: flex;
      gap: 0.75rem;
      margin-bottom: 1.5rem;
      max-width: 800px;
    }
    input[type="text"] {
      flex: 1;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 0.85rem 1.25rem;
      color: var(--text);
      font-size: 1rem;
      outline: none;
      transition: border-color 0.2s;
    }
    input[type="text"]:focus { border-color: var(--accent); }
    button {
      background: linear-gradient(135deg, var(--accent), #9b6af7);
      border: none;
      border-radius: 12px;
      color: white;
      padding: 0.85rem 1.75rem;
      font-size: 1rem;
      font-weight: 600;
      cursor: pointer;
      transition: transform 0.15s, box-shadow 0.15s;
    }
    button:hover { transform: translateY(-1px); box-shadow: 0 4px 20px rgba(124,106,247,0.4); }
    button:active { transform: translateY(0); }
    .examples { margin-bottom: 2rem; }
    .examples-label { color: var(--muted); font-size: 0.85rem; margin-bottom: 0.5rem; }
    .chip {
      display: inline-block;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 20px;
      padding: 0.3rem 0.85rem;
      font-size: 0.82rem;
      margin: 0.25rem;
      cursor: pointer;
      transition: border-color 0.2s, background 0.2s;
    }
    .chip:hover { border-color: var(--accent); background: rgba(124,106,247,0.1); }
    .status { color: var(--muted); font-size: 0.9rem; margin-bottom: 1.5rem; min-height: 1.2em; }
    .results-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
      gap: 1.25rem;
      max-width: 1200px;
    }
    .result-card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 16px;
      overflow: hidden;
      transition: transform 0.2s, box-shadow 0.2s;
    }
    .result-card:hover {
      transform: translateY(-4px);
      box-shadow: 0 8px 30px rgba(0,0,0,0.4);
    }
    .card-img-placeholder {
      background: linear-gradient(135deg, #1e1e2e, #2a1a3e);
      height: 200px;
      display: flex;
      align-items: center;
      justify-content: center;
      color: var(--muted);
      font-size: 0.8rem;
    }
    .card-img { width: 100%; height: 200px; object-fit: cover; display: block; }
    .card-body { padding: 1rem; }
    .card-rank {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 24px; height: 24px;
      background: var(--accent);
      border-radius: 50%;
      font-size: 0.75rem;
      font-weight: 700;
      margin-bottom: 0.5rem;
    }
    .score-bar {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      margin-bottom: 0.5rem;
    }
    .score-label { font-size: 0.75rem; color: var(--muted); width: 80px; }
    .bar-track {
      flex: 1;
      height: 4px;
      background: var(--border);
      border-radius: 2px;
      overflow: hidden;
    }
    .bar-fill { height: 100%; border-radius: 2px; }
    .score-val { font-size: 0.75rem; color: var(--muted); width: 40px; text-align: right; }
    .tags { display: flex; flex-wrap: wrap; gap: 0.3rem; margin-top: 0.6rem; }
    .tag {
      font-size: 0.72rem;
      padding: 0.15rem 0.5rem;
      border-radius: 10px;
      border: 1px solid var(--border);
      color: var(--muted);
    }
    .matched-attr {
      font-size: 0.7rem;
      color: #7cf77c;
      margin-top: 0.4rem;
    }
  </style>
</head>
<body>
  <h1>Glance Fashion Retrieval</h1>
  <p class="subtitle">Hybrid CLIP + BGE structured-caption search — understands what people wear, where they are, and the vibe of their outfit.</p>

  <div class="search-bar">
    <input type="text" id="query-input" placeholder="Try: A red tie and a white shirt in a formal setting." value="" />
    <button id="search-btn" onclick="runSearch()">Search</button>
  </div>

  <div class="examples">
    <div class="examples-label">Try an eval query:</div>
    <span class="chip" onclick="setQuery(this)">A person in a bright yellow raincoat.</span>
    <span class="chip" onclick="setQuery(this)">Professional business attire inside a modern office.</span>
    <span class="chip" onclick="setQuery(this)">Someone wearing a blue shirt sitting on a park bench.</span>
    <span class="chip" onclick="setQuery(this)">Casual weekend outfit for a city walk.</span>
    <span class="chip" onclick="setQuery(this)">A red tie and a white shirt in a formal setting.</span>
  </div>

  <div class="status" id="status">Ready — type a query above or click an example.</div>
  <div class="results-grid" id="results"></div>

  <script>
    function setQuery(el) {
      document.getElementById('query-input').value = el.textContent;
    }

    document.getElementById('query-input').addEventListener('keydown', e => {
      if (e.key === 'Enter') runSearch();
    });

    async function runSearch() {
      const query = document.getElementById('query-input').value.trim();
      if (!query) return;

      const status = document.getElementById('status');
      const resultsEl = document.getElementById('results');
      status.textContent = 'Searching...';
      resultsEl.innerHTML = '';

      try {
        const t0 = performance.now();
        const resp = await fetch('/search', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({query, top_k: 12})
        });
        const data = await resp.json();
        const elapsed = ((performance.now() - t0) / 1000).toFixed(2);

        status.textContent = `${data.total_results} results in ${elapsed}s — parse method: ${data.parsed_query.parse_method}`;
        renderResults(data.results);
      } catch (err) {
        status.textContent = 'Error: ' + err.message;
      }
    }

    function renderResults(results) {
      const container = document.getElementById('results');
      container.innerHTML = results.map((r, i) => {
        const meta = r.metadata || {};
        const clipPct = (r.clip_score * 100).toFixed(0);
        const textPct = (r.text_score * 100).toFixed(0);
        const totalPct = (r.fused_score * 100).toFixed(0);

        const imgTag = r.image_path
          ? `<img class="card-img" src="/image/${encodeURIComponent(r.image_id)}" alt="${r.image_id}" onerror="this.style.display='none';this.nextSibling.style.display='flex';" /><div class="card-img-placeholder" style="display:none">Image ${r.image_id}</div>`
          : `<div class="card-img-placeholder">Image ${r.image_id}</div>`;

        const tags = [
          meta.style, meta.environment, meta.dominant_color
        ].filter(Boolean).map(t => `<span class="tag">${t}</span>`).join('');

        const matched = (r.matched_attributes || [])
          .map(a => `Checkmark ${a}`).join(' · ');

        return `
          <div class="result-card">
            ${imgTag}
            <div class="card-body">
              <div class="card-rank">${i+1}</div>
              <div class="score-bar">
                <span class="score-label">Total</span>
                <div class="bar-track"><div class="bar-fill" style="width:${totalPct}%;background:linear-gradient(90deg,#7c6af7,#f77c6a)"></div></div>
                <span class="score-val">${(r.fused_score).toFixed(3)}</span>
              </div>
              <div class="score-bar">
                <span class="score-label">CLIP</span>
                <div class="bar-track"><div class="bar-fill" style="width:${clipPct}%;background:#7c6af7"></div></div>
                <span class="score-val">${(r.clip_score).toFixed(3)}</span>
              </div>
              <div class="score-bar">
                <span class="score-label">Text</span>
                <div class="bar-track"><div class="bar-fill" style="width:${textPct}%;background:#6af7b0"></div></div>
                <span class="score-val">${(r.text_score).toFixed(3)}</span>
              </div>
              <div class="tags">${tags}</div>
              ${matched ? `<div class="matched-attr">${matched}</div>` : ''}
            </div>
          </div>
        `;
      }).join('');
    }
  </script>
</body>
</html>"""
        return HTMLResponse(content=html)

    # --- Image serving ---
    @app.get("/image/{image_id}")
    async def serve_image(image_id: str):
        """Serve an indexed image by its ID (for the demo UI)."""
        from fastapi.responses import FileResponse
        retriever = app.state.retriever
        meta = retriever.vector_store.get_image_metadata(image_id)
        if not meta or not meta.get("path"):
            raise HTTPException(status_code=404, detail="Image not found")
        image_path = Path(meta["path"])
        if not image_path.exists():
            raise HTTPException(status_code=404, detail=f"Image file not found: {image_path}")
        return FileResponse(str(image_path))

    return app


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Part B Retriever — fashion image search CLI and API server.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single query
  python -m Part_B_Retriever.run_retrieval --query "A red tie and a white shirt"

  # Run all 5 official eval queries
  python -m Part_B_Retriever.run_retrieval --demo

  # Verbose output with score breakdown
  python -m Part_B_Retriever.run_retrieval --query "yellow raincoat" --verbose

  # Launch FastAPI server
  python -m Part_B_Retriever.run_retrieval --serve
        """
    )
    parser.add_argument("--query", type=str, default=None, help="Single search query.")
    parser.add_argument("--demo", action="store_true", help="Run all 5 official eval queries.")
    parser.add_argument("--serve", action="store_true", help="Launch FastAPI server.")
    parser.add_argument("--top_k", type=int, default=10, help="Number of results to return.")
    parser.add_argument("--verbose", action="store_true", help="Print full score breakdown.")
    parser.add_argument("--chroma_dir", type=Path, default=DEFAULT_CHROMA_DIR, help="ChromaDB directory.")
    parser.add_argument("--alpha", type=float, default=0.35, help="CLIP similarity weight.")
    parser.add_argument("--beta", type=float, default=0.50, help="BGE text similarity weight.")
    parser.add_argument("--gamma", type=float, default=0.15, help="Attribute bonus cap.")
    parser.add_argument("--use_rrf", action="store_true", help="Use Reciprocal Rank Fusion.")
    parser.add_argument("--host", type=str, default=os.getenv("API_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("API_PORT", "8000")))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.serve:
        import uvicorn
        app = create_fastapi_app(
            chroma_dir=args.chroma_dir,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
            use_rrf=args.use_rrf,
        )
        print(f"\n>> Starting Glance Fashion Retrieval API on http://{args.host}:{args.port}")
        print(f"   Demo UI:  http://localhost:{args.port}/")
        print(f"   API docs: http://localhost:{args.port}/docs\n")
        uvicorn.run(app, host=args.host, port=args.port)
        return

    retriever = build_retrieval_system(
        chroma_dir=args.chroma_dir,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        use_rrf=args.use_rrf,
    )

    if args.demo:
        run_demo(retriever, top_k=args.top_k, verbose=args.verbose)
    elif args.query:
        run_single_query(retriever, args.query, top_k=args.top_k, verbose=args.verbose)
    else:
        print("Usage: provide --query, --demo, or --serve. Run with -h for help.")
        sys.exit(1)


if __name__ == "__main__":
    main()
