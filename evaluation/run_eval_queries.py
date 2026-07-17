"""
evaluation/run_eval_queries.py
================================
Runs the 5 official evaluation queries against the built index, saves a
structured JSON report, and prints a human-readable summary table.

OUTPUT:
  - evaluation/results.json  : Full per-query results with all score breakdowns
  - stdout                   : Readable summary table for inclusion in submission PDF

DESIGN:
  This script is standalone — it imports from both Part_A and Part_B packages
  so it can run from any working directory as long as the repo root is in sys.path.
  The results JSON format is designed to be copy-paste-friendly for the submission PDF.
"""

import json
import sys
import time
from pathlib import Path

# Ensure repo root is on the Python path regardless of how this script is invoked
REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT))

# Force UTF-8 output on Windows (cp1252 default crashes on emoji).
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

from Part_B_Retriever.run_retrieval import build_retrieval_system, EVAL_QUERIES
from Part_B_Retriever.retriever import QueryExplainer

RESULTS_PATH = Path(__file__).parent / "results.json"
DEFAULT_CHROMA_DIR = REPO_ROOT / "chroma_db"
TOP_K = 10


def run_all_eval_queries(
    top_k: int = TOP_K,
    chroma_dir: Path = DEFAULT_CHROMA_DIR,
) -> dict:
    """
    Run all 5 eval queries and collect structured results.

    Returns a report dict suitable for JSON serialization.
    """
    retriever = build_retrieval_system(chroma_dir=chroma_dir)

    report = {
        "eval_queries": {},
        "summary": {},
        "metadata": {
            "top_k":       top_k,
            "fusion_weights": {
                "alpha": retriever.alpha,
                "beta":  retriever.beta,
                "gamma": retriever.gamma,
            },
            "use_rrf": retriever.use_rrf,
        },
    }

    print("\n" + "="*80)
    print("  OFFICIAL EVALUATION — Glance Fashion Retrieval")
    print("-"*80)

    # Table header
    print(f"\n{'Query':<60} {'#Results':>8} {'Top1 Score':>11} {'Latency':>8}")
    print("-" * 80)

    for query in EVAL_QUERIES:
        t0 = time.time()
        results, parsed = retriever.retrieve(query, top_k=top_k)
        latency_ms = (time.time() - t0) * 1000

        # Summary row
        top1_score = results[0].fused_score if results else 0.0
        print(
            f"{query[:58]:<60} "
            f"{len(results):>8} "
            f"{top1_score:>11.4f} "
            f"{latency_ms:>6.0f}ms"
        )

        # Detailed results for JSON
        report["eval_queries"][query] = {
            "parsed_query": parsed.to_dict(),
            "latency_ms":   round(latency_ms, 1),
            "results": [r.to_dict() for r in results],
            "top1_explanation": (
                QueryExplainer.explain(query, parsed, results[0])
                if results else "No results."
            ),
        }

    print("─"*80)
    print("\n--- Top-3 Results Per Query:\n")

    for query, data in report["eval_queries"].items():
        print(f"  Query: {query}")
        for i, r in enumerate(data["results"][:3], 1):
            meta = r["metadata"]
            print(
                f"    #{i}: {r['image_id']:<20}  fused={r['fused_score']:.4f}  "
                f"[{meta.get('style','?')[:10]:10s} | "
                f"{meta.get('environment','?')[:12]:12s} | "
                f"{meta.get('dominant_color','?')[:8]:8s}]"
            )
            if r["matched_attributes"]:
                print(f"       Matched: {', '.join(r['matched_attributes'])}")
        print()

    return report


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the 5 official evaluation queries.")
    parser.add_argument("--top_k", type=int, default=TOP_K, help="Results per query.")
    parser.add_argument(
        "--chroma_dir", type=Path, default=DEFAULT_CHROMA_DIR,
        help="ChromaDB directory.",
    )
    parser.add_argument(
        "--output", type=Path, default=RESULTS_PATH,
        help="Output path for results.json.",
    )
    args = parser.parse_args()

    report = run_all_eval_queries(top_k=args.top_k, chroma_dir=args.chroma_dir)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n[OK] Results saved to {args.output}")
    print(
        "   Include evaluation/results.json as an appendix in your submission PDF.\n"
    )


if __name__ == "__main__":
    main()
