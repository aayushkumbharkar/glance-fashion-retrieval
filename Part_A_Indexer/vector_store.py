"""
Part_A_Indexer/vector_store.py
================================
ChromaDB dual-collection manager for fashion image retrieval.

WHY TWO COLLECTIONS?
---------------------
We maintain two separate ChromaDB collections because CLIP embeddings (512-d)
and BGE caption embeddings (384-d) live in incompatible vector spaces:
  - `fashion_clip`  : 512-d cosine space. Indexes CLIP visual embeddings.
                      Similarity here captures visual aesthetics, color palette,
                      scene/setting recognition.
  - `fashion_text`  : 384-d cosine space. Indexes BGE embeddings of structured
                      caption text. Similarity here captures compositional attribute
                      matching — "red tie and white shirt" vs "white tie and red shirt"
                      are now distinguishable because the BGE text space can bind
                      color tokens to garment tokens via sentence-level context.

Sharing the same `image_id` key across both collections is what enables
score fusion at retrieval time: we can look up an image's CLIP score and
BGE score by ID and combine them.

SCALABILITY NOTE:
-----------------
ChromaDB uses HNSW (Hierarchical Navigable Small World) graphs under the hood,
which gives O(log N) approximate nearest-neighbor queries. This is fine up to
~1–2 million vectors per collection on commodity hardware.

Beyond that scale, the migration path is:
  1. Export embeddings from ChromaDB to numpy arrays.
  2. Build a FAISS IndexIVFPQ index:
     - IVF (Inverted File Index): divides the space into K Voronoi cells; query
       only visits nprobe cells (~1-5% of vectors) instead of all of them.
       This gives ~100x speedup at the cost of ~5% recall drop.
     - PQ (Product Quantization): compresses each 512-d float32 vector (~2 KB)
       into a 64-byte code, giving 32x memory compression. For 100M images,
       this is the difference between 200 GB and 6 GB of RAM.
     - Combined, IVF-PQ enables sub-millisecond queries at 100M+ scale with
       ~8x memory compression vs. brute-force float32. This is the standard
       production path for large-scale dense retrieval.
  3. Serve FAISS via a FAISS server or wrap it in a custom Python service.
  4. Metadata (for attribute filtering) would move to a columnar store like
     DuckDB or Parquet for efficient predicate pushdown.

ChromaDB → FAISS migration is a ~100-line operation (serialize, build index,
swap the query layer). This is why we don't prematurely optimize here.
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# ChromaDB collection names — keep these stable; changing them requires
# dropping and rebuilding the entire index.
CLIP_COLLECTION_NAME = "fashion_clip"
TEXT_COLLECTION_NAME = "fashion_text"

# Expected embedding dimensions (used for validation before insert)
CLIP_DIM = 512
TEXT_DIM = 384

# ChromaDB cosine distance space identifier
COSINE_SPACE = "cosine"


# ---------------------------------------------------------------------------
# Metadata schema helpers
# ---------------------------------------------------------------------------

def caption_to_metadata(caption_dict: dict, image_path: str) -> dict[str, Any]:
    """
    Convert a StructuredCaption dict to a flat ChromaDB-compatible metadata dict.

    ChromaDB requires metadata values to be scalars (str, int, float, bool).
    We join list fields with commas. Downstream code that reads metadata must
    split on commas to recover the lists — this convention is documented here
    so it's not implicit in the retriever.

    All values are lowercased for consistent metadata filtering (ChromaDB
    where clauses are case-sensitive by default, so we normalize at write time).
    """
    def to_str(val: Any) -> str:
        """Convert a value or list to a lowercase comma-joined string."""
        if isinstance(val, list):
            return ", ".join(v.lower() for v in val if v)
        return str(val).lower() if val else ""

    return {
        # Scalar enums — stored as-is (already validated to enum values)
        "environment":        str(caption_dict.get("environment", "unknown")).lower(),
        "style":              str(caption_dict.get("style", "unknown")).lower(),
        "dominant_color":     str(caption_dict.get("dominant_color", "unknown")).lower(),
        # List fields → comma-joined strings (ChromaDB doesn't support arrays)
        "clothing_items_str": to_str(caption_dict.get("clothing_items", [])),
        "colors_str":         to_str(caption_dict.get("colors", [])),
        "accessories_str":    to_str(caption_dict.get("accessories", [])),
        # File path — useful for UI to resolve images
        "path":               str(image_path),
    }


# ---------------------------------------------------------------------------
# FashionVectorStore — the core interface used by both indexer and retriever
# ---------------------------------------------------------------------------

class FashionVectorStore:
    """
    Manages two ChromaDB persistent collections for dual-space fashion retrieval.

    The client is persistent, meaning it writes to disk and survives process
    restarts. This is appropriate for an offline indexing pipeline.

    Args:
        persist_dir: Directory where ChromaDB stores its HNSW index and SQLite
                     metadata. Defaults to ./chroma_db relative to repo root.
    """

    def __init__(self, persist_dir: Optional[Path] = None):
        import chromadb

        if persist_dir is None:
            # Default to chroma_db/ at repo root (two levels up from this file)
            repo_root = Path(__file__).parent.parent.resolve()
            persist_dir = repo_root / "chroma_db"

        persist_dir.mkdir(parents=True, exist_ok=True)
        self.persist_dir = persist_dir

        logger.info(f"[FashionVectorStore] ChromaDB at {persist_dir}")

        # PersistentClient persists automatically; no explicit .persist() call needed
        # in ChromaDB >= 0.4.x (the old API required it).
        self._client = chromadb.PersistentClient(path=str(persist_dir))

        # get_or_create_collection is idempotent — safe to call on every startup.
        self._clip_collection = self._client.get_or_create_collection(
            name=CLIP_COLLECTION_NAME,
            metadata={"hnsw:space": COSINE_SPACE},
        )
        self._text_collection = self._client.get_or_create_collection(
            name=TEXT_COLLECTION_NAME,
            metadata={"hnsw:space": COSINE_SPACE},
        )

        logger.info(
            f"[FashionVectorStore] Collections ready — "
            f"CLIP: {self._clip_collection.count()} vectors, "
            f"Text: {self._text_collection.count()} vectors."
        )

    # --- Indexing ---

    def index_batch(
        self,
        image_ids: list[str],
        clip_embeddings: list[Optional[np.ndarray]],
        text_embeddings: list[Optional[np.ndarray]],
        metadata_list: list[dict],
        caption_texts: list[str],
    ) -> dict[str, int]:
        """
        Insert a batch of image embeddings into both collections, skipping IDs
        that are already indexed (idempotent re-runs).

        Args:
            image_ids:       List of unique image identifiers (filename stems).
            clip_embeddings: Parallel list of CLIP visual embeddings (512-d) or None.
            text_embeddings: Parallel list of BGE caption embeddings (384-d) or None.
            metadata_list:   Parallel list of flat metadata dicts.
            caption_texts:   Human-readable caption text (stored as ChromaDB document
                             for potential full-text search later).

        Returns:
            Dict with keys "clip_added", "text_added", "skipped".
        """
        # Determine which IDs are already in each collection
        existing_clip = self._get_existing_ids(self._clip_collection, image_ids)
        existing_text = self._get_existing_ids(self._text_collection, image_ids)

        clip_ids, clip_embs, clip_metas, clip_docs = [], [], [], []
        text_ids, text_embs, text_metas, text_docs = [], [], [], []

        for i, image_id in enumerate(image_ids):
            clip_emb = clip_embeddings[i]
            text_emb = text_embeddings[i]
            meta = metadata_list[i]
            doc_text = caption_texts[i] if i < len(caption_texts) else ""

            # Validate embedding dimensions before insert — a dimension mismatch
            # in ChromaDB raises a hard error that would abort the whole batch.
            if clip_emb is not None and image_id not in existing_clip:
                if clip_emb.shape[0] != CLIP_DIM:
                    logger.warning(
                        f"[FashionVectorStore] CLIP embedding dim mismatch for {image_id}: "
                        f"expected {CLIP_DIM}, got {clip_emb.shape[0]}. Skipping."
                    )
                else:
                    clip_ids.append(image_id)
                    clip_embs.append(clip_emb.tolist())
                    clip_metas.append(meta)
                    clip_docs.append(doc_text)

            if text_emb is not None and image_id not in existing_text:
                if text_emb.shape[0] != TEXT_DIM:
                    logger.warning(
                        f"[FashionVectorStore] Text embedding dim mismatch for {image_id}: "
                        f"expected {TEXT_DIM}, got {text_emb.shape[0]}. Skipping."
                    )
                else:
                    text_ids.append(image_id)
                    text_embs.append(text_emb.tolist())
                    text_metas.append(meta)
                    text_docs.append(doc_text)

        skipped = len(image_ids) - len(set(clip_ids) | set(text_ids))

        # ChromaDB upsert is atomic per call — if it fails partway, the items
        # that failed simply won't be in the index (no partial corruption).
        if clip_ids:
            self._clip_collection.upsert(
                ids=clip_ids,
                embeddings=clip_embs,
                metadatas=clip_metas,
                documents=clip_docs,
            )

        if text_ids:
            self._text_collection.upsert(
                ids=text_ids,
                embeddings=text_embs,
                metadatas=text_metas,
                documents=text_docs,
            )

        logger.info(
            f"[FashionVectorStore] Indexed {len(clip_ids)} CLIP, "
            f"{len(text_ids)} text embeddings. Skipped {skipped} existing."
        )
        return {
            "clip_added": len(clip_ids),
            "text_added": len(text_ids),
            "skipped": skipped,
        }

    # --- Retrieval ---

    def query_clip(
        self,
        query_embedding: np.ndarray,
        top_k: int = 20,
        where: Optional[dict] = None,
    ) -> list[dict[str, Any]]:
        """
        Query the CLIP collection and return normalized similarity scores.

        ChromaDB returns cosine *distance* (0=identical, 2=opposite). We convert
        to *similarity* via `sim = 1 - distance`, giving scores in [-1, 1] that
        are directly comparable to CLIP's own similarity conventions.

        Args:
            query_embedding: L2-normalized 512-d query vector.
            top_k:           Number of results to return.
            where:           Optional ChromaDB metadata filter dict.

        Returns:
            List of dicts: {id, similarity, metadata}.
        """
        return self._query_collection(
            self._clip_collection, query_embedding, top_k, where
        )

    def query_text(
        self,
        query_embedding: np.ndarray,
        top_k: int = 20,
        where: Optional[dict] = None,
    ) -> list[dict[str, Any]]:
        """
        Query the BGE text collection and return normalized similarity scores.

        Args:
            query_embedding: L2-normalized 384-d query vector.
            top_k:           Number of results to return.
            where:           Optional ChromaDB metadata filter dict.

        Returns:
            List of dicts: {id, similarity, metadata}.
        """
        return self._query_collection(
            self._text_collection, query_embedding, top_k, where
        )

    def _query_collection(
        self,
        collection,
        query_embedding: np.ndarray,
        top_k: int,
        where: Optional[dict],
    ) -> list[dict[str, Any]]:
        """Shared query logic for both collections."""
        query_kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding.tolist()],
            "n_results": min(top_k, collection.count()),
            "include": ["metadatas", "distances", "documents"],
        }
        if where:
            query_kwargs["where"] = where

        try:
            results = collection.query(**query_kwargs)
        except Exception as exc:
            logger.warning(f"[FashionVectorStore] Query failed: {exc}")
            return []

        if not results["ids"] or not results["ids"][0]:
            return []

        output = []
        for img_id, distance, meta, doc in zip(
            results["ids"][0],
            results["distances"][0],
            results["metadatas"][0],
            results["documents"][0],
        ):
            output.append({
                "id":         img_id,
                "similarity": float(1.0 - distance),  # cosine distance → similarity
                "metadata":   meta,
                "document":   doc,
            })
        return output

    # --- Utility ---

    def _get_existing_ids(self, collection, image_ids: list[str]) -> set[str]:
        """
        Check which image IDs are already in a collection.
        ChromaDB's .get() is the idiomatic way; it raises no error for missing IDs.
        """
        if not image_ids:
            return set()
        try:
            result = collection.get(ids=image_ids, include=[])
            return set(result["ids"])
        except Exception as exc:
            logger.warning(f"[FashionVectorStore] Could not check existing IDs: {exc}")
            return set()

    def get_collection_stats(self) -> dict[str, int]:
        """Return current vector counts for both collections."""
        return {
            "clip_vectors": self._clip_collection.count(),
            "text_vectors": self._text_collection.count(),
        }

    def get_image_metadata(self, image_id: str) -> Optional[dict]:
        """Retrieve stored metadata for a single image by ID."""
        try:
            result = self._clip_collection.get(ids=[image_id], include=["metadatas"])
            if result["metadatas"]:
                return result["metadatas"][0]
        except Exception as exc:
            logger.warning(f"[FashionVectorStore] Metadata lookup failed for {image_id}: {exc}")
        return None
