"""
Part_A_Indexer/feature_extractor.py
=====================================
Dual embedding engine: CLIP visual embeddings + sentence-transformer text embeddings.

WHY TWO SEPARATE EMBEDDING SPACES?
-----------------------------------
A single CLIP embedding collapses all image semantics into one 512-d vector.
For fashion retrieval this causes:
  1. Compositional binding failure: "red tie white shirt" ≈ "white tie red shirt"
     because CLIP's contrastive objective optimizes image↔text alignment globally,
     not locally (it can't bind "red" to "tie" specifically).
  2. Fashion vocabulary gap: CLIP was trained on web alt-text which underrepresents
     fashion-specific terms. Fine-grained retrieval ("blazer vs sport coat") fails.

By maintaining *two* embedding spaces:
  - CLIP space (512-d):   Holistic visual/aesthetic signal — handles "vibe", color
                           palette, setting recognition.
  - Text/BGE space (384-d): Embedding the *structured caption text* (prose generated
                            from our VLM-parsed JSON) puts compositional structure
                            into a retrieval-tuned sentence embedding space. BGE-small
                            was specifically trained on MS-MARCO / BEIR retrieval tasks,
                            so it handles attribute-based matching much better than
                            CLIP's text encoder.

WHY BGE-SMALL-EN-V1.5?
  BGE (BAAI General Embeddings) outperforms many larger models on MTEB retrieval
  benchmarks. The -v1.5 version uses a contrastive + knowledge-distillation training
  that makes it particularly good at dense passage retrieval — exactly our use case.
  "Small" was chosen over "large" for:
  - 5x faster inference (important for batch indexing)
  - Lower memory footprint for the vector store (384-d vs 1024-d)
  - Negligible retrieval quality gap on fashion-domain text

ASYMMETRIC PREFIXES (BGE convention):
  BGE recommends different prefixes for queries vs. documents to encode the
  asymmetric nature of retrieval (a short query ≠ a short document):
  - Document prefix: "Represent this sentence for retrieval: "
  - Query prefix:    "Represent this query for retrieving relevant passages: "
  These are from the official BGE model card on HuggingFace.

  NOTE: CLIP query embedding does NOT use the sentence transformer.
  CLIP has its own text encoder that was jointly trained with its visual encoder.
  Using the sentence transformer for the CLIP query would cross embedding spaces
  and produce meaningless similarities.
"""

import logging
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — keep these at module level so they're easy to audit/change.
# These defaults can be overridden via constructor args; nothing is hardcoded
# in the pipeline logic itself.
# ---------------------------------------------------------------------------

DEFAULT_CLIP_MODEL = "ViT-B-32"
DEFAULT_CLIP_PRETRAINED = "openai"   # "openai" weights are on par with LAION at B/32 scale;
                                      # swap to "laion2b_s34b_b79k" if higher recall is needed
DEFAULT_TEXT_MODEL = "BAAI/bge-small-en-v1.5"

# BGE asymmetric prefix strings (from official model card)
BGE_DOC_PREFIX = "Represent this sentence for retrieval: "
BGE_QUERY_PREFIX = "Represent this query for retrieving relevant passages: "

# CLIP vision produces 512-d embeddings at ViT-B-32 scale
CLIP_EMBED_DIM = 512
# BGE-small produces 384-d embeddings
BGE_EMBED_DIM = 384


# ---------------------------------------------------------------------------
# DualEmbedder — the core embedding interface used by run_indexer and retriever
# ---------------------------------------------------------------------------

class DualEmbedder:
    """
    Wraps OpenCLIP (visual + CLIP text) and SentenceTransformer (text/caption).

    Design decisions:
    - Constructor lazy-loads models to avoid slow startup when only one model
      is needed (e.g., caption-only mode).
    - All output embeddings are L2-normalized so that cosine similarity between
      any two embeddings reduces to a dot product. This is important for
      ChromaDB's HNSW index which works in cosine space.
    - Batch methods are provided for indexing throughput; per-item methods
      are provided for the query path (lower latency, simpler code).

    Args:
        clip_model_name:   OpenCLIP model architecture name.
        clip_pretrained:   OpenCLIP pretrained weights identifier.
        text_model_name:   SentenceTransformer model identifier.
        device:            'cuda', 'cpu', or None (auto-detect).
    """

    def __init__(
        self,
        clip_model_name: str = DEFAULT_CLIP_MODEL,
        clip_pretrained: str = DEFAULT_CLIP_PRETRAINED,
        text_model_name: str = DEFAULT_TEXT_MODEL,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model_name = clip_model_name
        self.clip_pretrained = clip_pretrained
        self.text_model_name = text_model_name

        self._clip_model = None
        self._clip_preprocess = None
        self._clip_tokenizer = None
        self._text_model = None

        logger.info(
            f"[DualEmbedder] Initialized (device={self.device}, "
            f"CLIP={clip_model_name}/{clip_pretrained}, text={text_model_name}). "
            "Models load lazily on first use."
        )

    # --- Lazy loaders ---

    def _ensure_clip(self) -> None:
        """Load OpenCLIP model and preprocessing pipeline if not already loaded."""
        if self._clip_model is not None:
            return
        import open_clip
        logger.info(f"[DualEmbedder] Loading OpenCLIP {self.clip_model_name}/{self.clip_pretrained}...")
        self._clip_model, _, self._clip_preprocess = open_clip.create_model_and_transforms(
            self.clip_model_name,
            pretrained=self.clip_pretrained,
        )
        self._clip_model = self._clip_model.to(self.device).eval()
        self._clip_tokenizer = open_clip.get_tokenizer(self.clip_model_name)
        logger.info(f"[DualEmbedder] CLIP loaded on {self.device}.")

    def _ensure_text_model(self) -> None:
        """Load SentenceTransformer model if not already loaded."""
        if self._text_model is not None:
            return
        from sentence_transformers import SentenceTransformer
        logger.info(f"[DualEmbedder] Loading SentenceTransformer {self.text_model_name}...")
        self._text_model = SentenceTransformer(self.text_model_name, device=self.device)
        logger.info(f"[DualEmbedder] SentenceTransformer loaded.")

    # --- Image embedding (CLIP visual encoder) ---

    def embed_image(self, image_path: "Path") -> Optional[np.ndarray]:
        """
        Embed a single image using CLIP's visual encoder.

        Returns:
            L2-normalized float32 ndarray of shape (512,), or None on failure.
        """
        self._ensure_clip()
        from PIL import Image
        try:
            image = Image.open(image_path).convert("RGB")
            tensor = self._clip_preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                embedding = self._clip_model.encode_image(tensor)
            return self._l2_normalize(embedding.cpu().numpy()[0])
        except Exception as exc:
            logger.warning(f"[DualEmbedder] Failed to embed image {image_path}: {exc}")
            return None

    def embed_images_batch(
        self, image_paths: list, batch_size: int = 32
    ) -> dict[str, Optional[np.ndarray]]:
        """
        Batch embed images for indexing throughput.

        Batching is critical here: CLIP's ViT processes images in parallel on GPU.
        With batch_size=32 on an RTX 3090, we process ~20 images/sec;
        single-image mode would be ~3 images/sec.

        Args:
            image_paths: List of Path objects.
            batch_size:  Images per GPU forward pass.

        Returns:
            Dict mapping image_id (Path.stem) → normalized embedding (or None).
        """
        from pathlib import Path
        from PIL import Image
        from tqdm import tqdm

        self._ensure_clip()
        results: dict[str, Optional[np.ndarray]] = {}

        for i in tqdm(range(0, len(image_paths), batch_size), desc="Embedding images (CLIP)"):
            batch_paths = image_paths[i:i + batch_size]
            tensors = []
            valid_ids = []

            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    tensors.append(self._clip_preprocess(img))
                    valid_ids.append(Path(p).stem)
                except Exception as exc:
                    logger.warning(f"[DualEmbedder] Skipping {p}: {exc}")
                    results[Path(p).stem] = None

            if not tensors:
                continue

            batch_tensor = torch.stack(tensors).to(self.device)
            with torch.no_grad():
                embeddings = self._clip_model.encode_image(batch_tensor)
            embeddings = embeddings.cpu().numpy()

            for img_id, emb in zip(valid_ids, embeddings):
                results[img_id] = self._l2_normalize(emb)

        return results

    # --- Caption/text embedding (sentence transformer, BGE) ---

    def embed_caption(self, caption_text: str) -> np.ndarray:
        """
        Embed a caption/document using BGE with the document prefix.

        The document prefix is applied here (not by the caller) to ensure
        the prefix convention is always enforced correctly. If a caller
        accidentally embeds a query as a document, the prefix mismatch would
        silently degrade retrieval quality.

        Returns:
            L2-normalized float32 ndarray of shape (384,).
        """
        self._ensure_text_model()
        prefixed = BGE_DOC_PREFIX + caption_text
        embedding = self._text_model.encode(prefixed, normalize_embeddings=True)
        return embedding.astype(np.float32)

    def embed_captions_batch(
        self, caption_texts: list[str], batch_size: int = 64
    ) -> np.ndarray:
        """
        Batch embed caption texts for indexing.

        Returns:
            ndarray of shape (N, 384), L2-normalized.
        """
        self._ensure_text_model()
        prefixed = [BGE_DOC_PREFIX + t for t in caption_texts]
        embeddings = self._text_model.encode(
            prefixed,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return embeddings.astype(np.float32)

    # --- Query embedding (BOTH spaces, for hybrid retrieval) ---

    def embed_query_clip(self, query_text: str) -> Optional[np.ndarray]:
        """
        Embed a text query using CLIP's *own* text encoder.

        WHY CLIP's OWN TEXT ENCODER HERE?
        CLIP's visual and text encoders were jointly trained so that
        [image embedding] ≈ [text embedding] in the same 512-d space.
        If we used BGE to embed the query for CLIP-space retrieval,
        we'd be comparing BGE-encoded queries against CLIP-encoded images —
        which are in completely different spaces and would produce garbage similarities.

        Returns:
            L2-normalized ndarray of shape (512,), or None on failure.
        """
        self._ensure_clip()
        try:
            tokens = self._clip_tokenizer([query_text]).to(self.device)
            with torch.no_grad():
                text_features = self._clip_model.encode_text(tokens)
            return self._l2_normalize(text_features.cpu().numpy()[0])
        except Exception as exc:
            logger.warning(f"[DualEmbedder] CLIP text embedding failed for query: {exc}")
            return None

    def embed_query_text(self, query_text: str) -> np.ndarray:
        """
        Embed a text query using BGE with the query prefix.

        The query prefix tells the BGE model this is a short asymmetric query,
        not a document-length passage. This asymmetric design was shown in the
        BGE paper to improve NDCG on retrieval benchmarks vs. using the same
        prefix for both.

        Returns:
            L2-normalized ndarray of shape (384,).
        """
        self._ensure_text_model()
        prefixed = BGE_QUERY_PREFIX + query_text
        embedding = self._text_model.encode(prefixed, normalize_embeddings=True)
        return embedding.astype(np.float32)

    # --- Utility ---

    @staticmethod
    def _l2_normalize(vec: np.ndarray) -> np.ndarray:
        """
        L2-normalize a vector. Safe against zero-norm edge case (returns zeros
        rather than NaN, which would corrupt distance computations in ChromaDB).
        """
        norm = np.linalg.norm(vec)
        if norm < 1e-10:
            logger.warning("[DualEmbedder] Near-zero norm encountered; returning zero vector.")
            return vec.astype(np.float32)
        return (vec / norm).astype(np.float32)
