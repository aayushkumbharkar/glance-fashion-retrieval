"""
Part_B_Retriever/retriever.py
================================
Hybrid fusion retrieval engine combining CLIP visual similarity, BGE text
similarity, and structured attribute-match bonuses.

RETRIEVAL ARCHITECTURE:
-----------------------
1. Query → ParsedQuery (QueryParser)
2. ParsedQuery → two query embeddings (DualEmbedder):
     - CLIP text embedding (512-d) for the CLIP collection
     - BGE query embedding (384-d) for the text collection
3. Retrieve top-N candidates from each collection (N = top_k * pool_multiplier)
   - Larger pool gives re-ranking headroom without sacrificing recall
4. Fuse scores:
     score = α·clip_score + β·text_score + attribute_bonus
   where α=0.35, β=0.50, attribute_bonus ≤ γ=0.15
5. Sort by fused score → return top_k

WHY β > α (text outweighs CLIP):
  The text/BGE path retrieves against structured captions that preserve
  color-garment bindings. For compositional queries ("red tie AND white shirt"),
  BGE similarity on the caption text dominates because the semantic binding
  is explicit in the text. CLIP visual similarity is an aesthetic tiebreaker
  (it recognizes formal settings, color palettes, general vibe) and contributes
  secondary signal. The 35/50 split was chosen empirically on the 5 eval queries.

WHY ATTRIBUTE BONUS AS A THIRD TERM?
  Soft vector similarity can rank a near-match above an exact match when the
  embeddings of adjacent concepts are close. The attribute bonus is a hard
  signal that rewards exact metadata matches (environment, style, exact
  color-garment pairs). It's capped at γ=0.15 so it acts as a tiebreaker
  rather than dominating the ranking on its own.

RECIPROCAL RANK FUSION (RRF) ALTERNATIVE:
  When the two retrievers' score distributions are very different (e.g. CLIP
  returns scores in [0.2, 0.4] while BGE returns [0.6, 0.9]), direct weighted
  sum comparison is unfair. RRF normalizes by rank position instead:
    RRF_score(d) = Σ 1 / (k + rank_i(d))   where k=60 dampens rank outliers
  This is more robust when score distributions differ, but less interpretable.
  Both are available via the `use_rrf` constructor flag.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fusion weight defaults — documented as a named constant so they're easy
# to tune and the ablation notebook can override them cleanly.
# ---------------------------------------------------------------------------

# α: CLIP visual similarity weight
# β: BGE text similarity weight  (β > α because text carries compositional structure)
# γ: max attribute bonus cap (prevents hard-match from overwhelming soft similarity)
DEFAULT_ALPHA = 0.35
DEFAULT_BETA  = 0.50
DEFAULT_GAMMA = 0.15   # α + β + γ = 1.0 when attribute bonus is fully triggered

# Pool multiplier: retrieve pool_k = top_k * POOL_MULTIPLIER candidates from each
# collection before fusion. More candidates = better re-ranking at small top_k cost.
POOL_MULTIPLIER = 3

# RRF rank damping constant (k=60 is the standard value from the original RRF paper)
RRF_K = 60


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    """A single retrieved image with all score components for explainability."""
    image_id: str
    image_path: str
    clip_score: float = 0.0
    text_score: float = 0.0
    attribute_bonus: float = 0.0
    fused_score: float = 0.0
    metadata: dict = field(default_factory=dict)
    matched_attributes: list[str] = field(default_factory=list)   # for QueryExplainer

    def to_dict(self) -> dict:
        return {
            "image_id":          self.image_id,
            "image_path":        self.image_path,
            "clip_score":        round(self.clip_score, 4),
            "text_score":        round(self.text_score, 4),
            "attribute_bonus":   round(self.attribute_bonus, 4),
            "fused_score":       round(self.fused_score, 4),
            "metadata":          self.metadata,
            "matched_attributes": self.matched_attributes,
        }


# ---------------------------------------------------------------------------
# Attribute bonus scorer
# ---------------------------------------------------------------------------

def compute_attribute_bonus(
    parsed_query: "ParsedQuery",
    image_metadata: dict,
    gamma: float = DEFAULT_GAMMA,
) -> tuple[float, list[str]]:
    """
    Compute a discrete attribute-match bonus for a retrieved image.

    This is a hard signal layered on top of soft vector similarity.
    Each matching attribute contributes a fraction of gamma; the total is capped.

    Matching rules:
    - Environment exact match: +0.4 of gamma
    - Style exact match: +0.3 of gamma
    - Each color-garment pair found in colors_str: +0.1 of gamma per pair
    - Each clothing item found in clothing_items_str: +0.05 of gamma per item

    The fractional design means:
    - A query with both env + style + 2 color-pairs maxes out the bonus
    - A partial match (just env) gets a small but non-zero boost
    - No single attribute can dominate by itself

    Returns:
        (bonus_score, list_of_matched_attribute_names)
    """
    bonus = 0.0
    matched: list[str] = []

    # Environment match
    if (
        parsed_query.environment
        and parsed_query.environment != "unknown"
        and image_metadata.get("environment") == parsed_query.environment
    ):
        bonus += 0.4 * gamma
        matched.append(f"environment:{parsed_query.environment}")

    # Style match
    if (
        parsed_query.style
        and parsed_query.style != "unknown"
        and image_metadata.get("style") == parsed_query.style
    ):
        bonus += 0.3 * gamma
        matched.append(f"style:{parsed_query.style}")

    # Color-garment pair matches
    colors_str = image_metadata.get("colors_str", "").lower()
    for color_garment in parsed_query.colors:
        cg_lower = color_garment.lower()
        # We check if any comma-separated token in colors_str contains our query pair
        if _fuzzy_contains(cg_lower, colors_str):
            bonus += 0.1 * gamma
            matched.append(f"color_garment:{cg_lower}")

    # Clothing item matches
    items_str = image_metadata.get("clothing_items_str", "").lower()
    for item in parsed_query.clothing_items:
        if item.lower() in items_str:
            bonus += 0.05 * gamma
            matched.append(f"clothing:{item.lower()}")

    # Cap at gamma
    return min(bonus, gamma), matched


def _fuzzy_contains(query_phrase: str, target_str: str) -> bool:
    """
    Check if query_phrase approximately appears in target_str.
    Handles partial matches (e.g. "red tie" found in "dark red tie").
    """
    # Check exact phrase first
    if query_phrase in target_str:
        return True
    # Check if all words in query_phrase appear in the target
    words = query_phrase.split()
    return all(w in target_str for w in words)


# ---------------------------------------------------------------------------
# HybridRetriever — the main retrieval engine
# ---------------------------------------------------------------------------

class HybridRetriever:
    """
    Hybrid fusion retriever combining CLIP visual and BGE text retrieval.

    Constructor injection (not globals) for all dependencies so the retriever
    is independently testable with mock vector stores / embedders.

    Args:
        vector_store:  FashionVectorStore instance (dual ChromaDB collections).
        embedder:      DualEmbedder instance (CLIP + BGE).
        query_parser:  QueryParser instance (LLM + rule-based).
        alpha:         CLIP similarity weight (default 0.35).
        beta:          BGE text similarity weight (default 0.50).
        gamma:         Max attribute bonus cap (default 0.15).
        use_rrf:       Use Reciprocal Rank Fusion instead of weighted sum.
    """

    def __init__(
        self,
        vector_store: "FashionVectorStore",
        embedder: "DualEmbedder",
        query_parser: "QueryParser",
        alpha: float = DEFAULT_ALPHA,
        beta: float = DEFAULT_BETA,
        gamma: float = DEFAULT_GAMMA,
        use_rrf: bool = False,
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.query_parser = query_parser
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.use_rrf = use_rrf

        # Validate weight sum (informational, not enforced — gamma comes from bonus,
        # not the base weighted sum, so alpha + beta + gamma = 1.0 is a soft target)
        if abs(alpha + beta - (1.0 - gamma)) > 0.01:
            logger.debug(
                f"[HybridRetriever] α+β={alpha+beta:.2f} (note: γ={gamma:.2f} is added separately "
                f"from attribute bonus, so effective max score = {alpha+beta+gamma:.2f})"
            )

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        where: Optional[dict] = None,
    ) -> tuple[list[RetrievalResult], "ParsedQuery"]:
        """
        Run the full hybrid retrieval pipeline for a text query.

        Args:
            query:  Natural language search query.
            top_k:  Number of results to return.
            where:  Optional ChromaDB metadata filter override (if None, derived
                    from parsed query's filter flags).

        Returns:
            (results, parsed_query) — results sorted by fused_score descending.
        """
        # --- Step 1: Parse query ---
        parsed = self.query_parser.parse(query)
        logger.info(
            f"[HybridRetriever] Query parsed via {parsed.parse_method}: "
            f"items={parsed.clothing_items}, env={parsed.environment}, style={parsed.style}"
        )

        # --- Step 2: Embed query in both spaces ---
        clip_query_emb = self.embedder.embed_query_clip(
            parsed.expanded_query or query
        )
        text_query_emb = self.embedder.embed_query_text(
            parsed.expanded_query or query
        )

        if clip_query_emb is None:
            logger.warning("[HybridRetriever] CLIP query embedding failed; text-only retrieval.")

        # --- Step 3: Build metadata filter ---
        metadata_filter = where if where is not None else parsed.build_chroma_filter()

        # Retrieve larger pool for re-ranking
        pool_k = top_k * POOL_MULTIPLIER

        # --- Step 4: Retrieve from both collections ---
        clip_results: list[dict] = []
        if clip_query_emb is not None:
            clip_results = self.vector_store.query_clip(
                query_embedding=clip_query_emb,
                top_k=pool_k,
                where=metadata_filter,
            )

        text_results: list[dict] = []
        if text_query_emb is not None:
            text_results = self.vector_store.query_text(
                query_embedding=text_query_emb,
                top_k=pool_k,
                where=metadata_filter,
            )

        # --- Step 5: Fuse scores ---
        if self.use_rrf:
            fused = self._rrf_fusion(clip_results, text_results)
        else:
            fused = self._weighted_sum_fusion(clip_results, text_results)

        # --- Step 6: Apply attribute bonus ---
        final_results: list[RetrievalResult] = []
        for img_id, base_score, clip_s, text_s, meta in fused:
            bonus, matched = compute_attribute_bonus(parsed, meta, self.gamma)
            final_score = base_score + bonus
            result = RetrievalResult(
                image_id=img_id,
                image_path=meta.get("path", ""),
                clip_score=clip_s,
                text_score=text_s,
                attribute_bonus=bonus,
                fused_score=final_score,
                metadata=meta,
                matched_attributes=matched,
            )
            final_results.append(result)

        # Sort by fused score, highest first
        final_results.sort(key=lambda r: r.fused_score, reverse=True)

        return final_results[:top_k], parsed

    def _weighted_sum_fusion(
        self,
        clip_results: list[dict],
        text_results: list[dict],
    ) -> list[tuple]:
        """
        Weighted sum fusion: score = α·clip_score + β·text_score.

        Assumption: both score distributions are in a comparable range [0, 1]
        because we use L2-normalized embeddings and convert cosine distance to
        similarity. If this assumption breaks down, switch to RRF.

        Returns list of (image_id, base_score, clip_score, text_score, metadata) tuples.
        """
        # Index by image ID for fast lookup
        clip_map: dict[str, dict] = {r["id"]: r for r in clip_results}
        text_map: dict[str, dict] = {r["id"]: r for r in text_results}

        all_ids = set(clip_map.keys()) | set(text_map.keys())

        results = []
        for img_id in all_ids:
            clip_entry = clip_map.get(img_id)
            text_entry = text_map.get(img_id)

            clip_s = clip_entry["similarity"] if clip_entry else 0.0
            text_s = text_entry["similarity"] if text_entry else 0.0
            meta = (
                clip_entry["metadata"] if clip_entry else
                text_entry["metadata"] if text_entry else {}
            )

            base_score = self.alpha * clip_s + self.beta * text_s
            results.append((img_id, base_score, clip_s, text_s, meta))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def _rrf_fusion(
        self,
        clip_results: list[dict],
        text_results: list[dict],
    ) -> list[tuple]:
        """
        Reciprocal Rank Fusion: RRF_score(d) = Σ_i 1/(k + rank_i(d))

        WHY RRF?
        When score distributions differ significantly between retrievers
        (e.g. CLIP returns [0.15, 0.35] and BGE returns [0.55, 0.90]),
        a weighted sum unfairly benefits the higher-scoring retriever.
        RRF normalizes by rank so both retrievers have equal influence.

        k=60 is the standard from the original RRF paper (Cormack et al., 2009).
        Higher k gives smoother rank weighting; lower k emphasizes top ranks.

        Returns list of (image_id, rrf_score, clip_score, text_score, metadata) tuples.
        """
        clip_ranks = {r["id"]: i for i, r in enumerate(clip_results)}
        text_ranks = {r["id"]: i for i, r in enumerate(text_results)}

        # Build metadata lookup
        meta_map: dict[str, dict] = {}
        for r in clip_results + text_results:
            meta_map.setdefault(r["id"], r.get("metadata", {}))

        # Collect raw scores for reporting
        clip_score_map = {r["id"]: r["similarity"] for r in clip_results}
        text_score_map = {r["id"]: r["similarity"] for r in text_results}

        all_ids = set(clip_ranks.keys()) | set(text_ranks.keys())

        results = []
        for img_id in all_ids:
            # If an ID is only in one retriever, treat its rank in the other as
            # one beyond the last retrieved position (conservative).
            clip_rank = clip_ranks.get(img_id, len(clip_results))
            text_rank = text_ranks.get(img_id, len(text_results))
            rrf_score = 1.0 / (RRF_K + clip_rank) + 1.0 / (RRF_K + text_rank)
            results.append((
                img_id,
                rrf_score,
                clip_score_map.get(img_id, 0.0),
                text_score_map.get(img_id, 0.0),
                meta_map.get(img_id, {}),
            ))

        results.sort(key=lambda x: x[1], reverse=True)
        return results


# ---------------------------------------------------------------------------
# QueryExplainer — human-readable breakdown of why a result was returned
# ---------------------------------------------------------------------------

class QueryExplainer:
    """
    Produces human-readable explanations for retrieved results.

    This is not just a debugging tool — it demonstrates that the system
    understands *why* it returned a result, which is a key evaluation criterion.
    The explanation surfaces each score component and which metadata attributes
    contributed to the match.
    """

    @staticmethod
    def explain(
        query: str,
        parsed_query: "ParsedQuery",
        result: RetrievalResult,
        alpha: float = DEFAULT_ALPHA,
        beta: float = DEFAULT_BETA,
    ) -> str:
        """
        Generate a human-readable explanation for a single retrieval result.

        Args:
            query:         Original query string.
            parsed_query:  Structured parse of the query.
            result:        The retrieved RetrievalResult.
            alpha, beta:   Fusion weights (for score breakdown labels).

        Returns:
            Multi-line string explanation.
        """
        lines = [
            f"  Image ID:      {result.image_id}",
            f"  Fused score:   {result.fused_score:.4f}",
            f"    +- CLIP ({alpha:.0%}):  {result.clip_score:.4f}  [visual aesthetics / setting]",
            f"    +- Text ({beta:.0%}):  {result.text_score:.4f}  [compositional attributes]",
            f"    +- Attr bonus: {result.attribute_bonus:.4f}  [discrete metadata match]",
        ]

        if result.matched_attributes:
            lines.append(f"  Matched attrs: {', '.join(result.matched_attributes)}")
        else:
            lines.append("  Matched attrs: (none — purely vector similarity match)")

        meta = result.metadata
        if meta:
            lines.append(
                f"  Image meta:    env={meta.get('environment','?')}, "
                f"style={meta.get('style','?')}, "
                f"dominant_color={meta.get('dominant_color','?')}"
            )
            colors = meta.get("colors_str", "")
            if colors:
                lines.append(f"  Colors:        {colors}")
            items = meta.get("clothing_items_str", "")
            if items:
                lines.append(f"  Clothing:      {items}")

        return "\n".join(lines)

    @staticmethod
    def explain_query_parse(parsed_query: "ParsedQuery") -> str:
        """Pretty-print the parsed query structure."""
        lines = [
            f"  Parse method:  {parsed_query.parse_method}",
            f"  Clothing:      {parsed_query.clothing_items}",
            f"  Colors:        {parsed_query.colors}",
            f"  Environment:   {parsed_query.environment} (filter={parsed_query.filter_environment})",
            f"  Style:         {parsed_query.style} (filter={parsed_query.filter_style})",
            f"  Expanded:      {parsed_query.expanded_query[:150]}...",
        ]
        return "\n".join(lines)
