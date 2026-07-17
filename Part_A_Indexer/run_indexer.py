"""
Part_A_Indexer/run_indexer.py
==============================
CLI entrypoint for the indexing pipeline (Part A).

Pipeline flow:
  1. Load image list from metadata.json (or scan image_dir directly)
  2. Generate structured captions via VLM (BLIP-2 or BLIP base)
  3. Compute dual embeddings: CLIP visual + BGE caption text
  4. Insert into ChromaDB dual collections
  5. Print summary statistics

RESUMABILITY:
  Each stage is independently resumable:
  - Caption stage: checkpoint file (data/captions_checkpoint.json)
    stores completed image IDs; re-running skips them.
  - Embedding/indexing stage: ChromaDB upsert is idempotent — IDs already
    in the collection are skipped without error.
  This means a crash at any stage loses at most `--checkpoint_every` captions.

MODES:
  --mode full         Run all three stages (caption, embed, index)
  --mode caption_only Stop after generating captions (useful for testing VLM quality)
  --mode embed_only   Load existing captions and run embed+index (assumes captions done)
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

# Force UTF-8 output on Windows (cp1252 default console encoding crashes on
# emoji / special chars in print statements).
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Load .env before anything else so model names and paths resolve correctly
from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Configure logging early so all module-level loggers capture to stdout
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path resolution — everything is relative to the repository root
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent.resolve()
DEFAULT_IMAGE_DIR   = REPO_ROOT / "data" / "images"
DEFAULT_METADATA    = REPO_ROOT / "data" / "metadata.json"
DEFAULT_CAPTIONS    = REPO_ROOT / "data" / "captions.json"
DEFAULT_CHECKPOINT  = REPO_ROOT / "data" / "captions_checkpoint.json"
DEFAULT_CHROMA_DIR  = REPO_ROOT / "chroma_db"


# ---------------------------------------------------------------------------
# Image list loading
# ---------------------------------------------------------------------------

def load_image_list(
    metadata_path: Path,
    image_dir: Path,
    limit: Optional[int] = None,
) -> list[Path]:
    """
    Resolve the list of images to index.

    Priority:
      1. metadata.json (created by download_dataset.py) — preferred because
         it represents the stratified/sampled subset.
      2. Direct directory scan — fallback when metadata.json doesn't exist.
    """
    if metadata_path.exists():
        with open(metadata_path) as f:
            meta = json.load(f)
        paths = [Path(item["path"]) for item in meta.get("images", [])]
        logger.info(f"[Indexer] Loaded {len(paths)} images from {metadata_path}.")
    elif image_dir.exists():
        logger.warning(
            f"[Indexer] metadata.json not found; scanning {image_dir} directly. "
            "Consider running download_dataset.py first for reproducible subset."
        )
        paths = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
        logger.info(f"[Indexer] Found {len(paths)} images in {image_dir}.")
    else:
        logger.error(
            f"[Indexer] Neither {metadata_path} nor {image_dir} exist. "
            "Run data/download_dataset.py first."
        )
        sys.exit(1)

    if limit:
        paths = paths[:limit]
        logger.info(f"[Indexer] Limiting to {len(paths)} images (--limit={limit}).")

    return paths


# ---------------------------------------------------------------------------
# Caption stage
# ---------------------------------------------------------------------------

def run_caption_stage(
    image_paths: list[Path],
    captions_file: Path,
    checkpoint_path: Path,
    use_lightweight: bool,
    use_4bit: bool,
    checkpoint_every: int = 50,
) -> dict[str, dict]:
    """
    Run the VLM caption generation stage.

    Loads existing captions from captions_file if present (merge, not overwrite).
    Returns the full mapping of image_id → caption_dict.
    """
    # Load already-generated captions
    existing_captions: dict[str, dict] = {}
    if captions_file.exists():
        with open(captions_file) as f:
            existing_captions = json.load(f)
        logger.info(f"[Indexer] Loaded {len(existing_captions)} existing captions.")

    # Determine which images still need captions
    needs_caption = [p for p in image_paths if p.stem not in existing_captions]
    logger.info(f"[Indexer] {len(needs_caption)} images need captions.")

    if not needs_caption:
        return existing_captions

    # Load the appropriate VLM
    if use_lightweight:
        from Part_A_Indexer.caption_generator import LightweightCaptionGenerator
        generator = LightweightCaptionGenerator(checkpoint_path=checkpoint_path)
        logger.info("[Indexer] Using LightweightCaptionGenerator (BLIP base).")
    else:
        from Part_A_Indexer.caption_generator import CaptionGenerator
        generator = CaptionGenerator(
            use_4bit=use_4bit,
            checkpoint_path=checkpoint_path,
        )
        logger.info("[Indexer] Using CaptionGenerator (BLIP-2).")

    new_captions = generator.generate_batch(needs_caption, checkpoint_every=checkpoint_every)

    # Merge and save
    for img_id, caption in new_captions.items():
        existing_captions[img_id] = caption.to_dict()

    captions_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = captions_file.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(existing_captions, f, indent=2)
    tmp.replace(captions_file)
    logger.info(f"[Indexer] Saved {len(existing_captions)} captions to {captions_file}.")

    return existing_captions


# ---------------------------------------------------------------------------
# Embed + index stage
# ---------------------------------------------------------------------------

def run_embed_index_stage(
    image_paths: list[Path],
    captions: dict[str, dict],
    chroma_dir: Path,
    clip_model: str,
    clip_pretrained: str,
    text_model: str,
    batch_size: int = 32,
) -> dict[str, int]:
    """
    Compute dual embeddings and insert into ChromaDB.

    Stages:
      1. Embed images (CLIP visual) — GPU-accelerated batch
      2. Embed captions (BGE text) — GPU-accelerated batch
      3. Insert both into ChromaDB with metadata
    """
    from Part_A_Indexer.feature_extractor import DualEmbedder
    from Part_A_Indexer.vector_store import FashionVectorStore, caption_to_metadata
    from Part_A_Indexer.caption_generator import caption_to_text, StructuredCaption

    embedder = DualEmbedder(
        clip_model_name=clip_model,
        clip_pretrained=clip_pretrained,
        text_model_name=text_model,
    )
    vector_store = FashionVectorStore(persist_dir=chroma_dir)

    # Determine which images need embedding (not yet in CLIP collection)
    existing_stats = vector_store.get_collection_stats()
    logger.info(
        f"[Indexer] Vector store has {existing_stats['clip_vectors']} CLIP, "
        f"{existing_stats['text_vectors']} text vectors."
    )

    # --- Step 1: CLIP visual embeddings ---
    logger.info(f"[Indexer] Computing CLIP embeddings for {len(image_paths)} images...")
    clip_embeddings_map = embedder.embed_images_batch(image_paths, batch_size=batch_size)

    # --- Step 2: BGE caption embeddings ---
    # Build caption texts for images that have captions
    ids_to_embed = [p.stem for p in image_paths if p.stem in captions]
    caption_texts_to_embed = []
    for img_id in ids_to_embed:
        cap_dict = captions[img_id]
        cap_obj = StructuredCaption.from_dict(cap_dict)
        caption_texts_to_embed.append(caption_to_text(cap_obj))

    logger.info(f"[Indexer] Computing BGE text embeddings for {len(ids_to_embed)} captions...")
    text_embeddings_arr = embedder.embed_captions_batch(caption_texts_to_embed, batch_size=64)

    text_embeddings_map = {
        img_id: text_embeddings_arr[i]
        for i, img_id in enumerate(ids_to_embed)
    }

    # --- Step 3: Prepare batch for insertion ---
    all_ids = [p.stem for p in image_paths]
    clip_embs_list = [clip_embeddings_map.get(img_id) for img_id in all_ids]
    text_embs_list = [text_embeddings_map.get(img_id) for img_id in all_ids]

    metadata_list = []
    caption_texts_list = []
    for img_id, path in zip(all_ids, image_paths):
        cap_dict = captions.get(img_id, {})
        meta = caption_to_metadata(cap_dict, str(path.resolve()))
        cap_obj = StructuredCaption.from_dict(cap_dict) if cap_dict else StructuredCaption()
        text = caption_to_text(cap_obj)
        metadata_list.append(meta)
        caption_texts_list.append(text)

    # --- Step 4: Insert ---
    stats = vector_store.index_batch(
        image_ids=all_ids,
        clip_embeddings=clip_embs_list,
        text_embeddings=text_embs_list,
        metadata_list=metadata_list,
        caption_texts=caption_texts_list,
    )

    # Final counts
    final_stats = vector_store.get_collection_stats()
    logger.info(
        f"[Indexer] Final vector store: {final_stats['clip_vectors']} CLIP, "
        f"{final_stats['text_vectors']} text vectors."
    )
    return final_stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Part A Indexer — generates structured captions and builds the "
            "dual-space vector store for fashion image retrieval."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline (caption + embed + index)
  python -m Part_A_Indexer.run_indexer --mode full

  # Caption only (inspect VLM output before committing to index)
  python -m Part_A_Indexer.run_indexer --mode caption_only --limit 10

  # Embed + index using pre-generated captions
  python -m Part_A_Indexer.run_indexer --mode embed_only

  # Use lightweight BLIP base (CPU-friendly)
  python -m Part_A_Indexer.run_indexer --use_lightweight_vlm
        """
    )
    parser.add_argument(
        "--image_dir", type=Path, default=DEFAULT_IMAGE_DIR,
        help="Directory containing images to index.",
    )
    parser.add_argument(
        "--metadata_path", type=Path, default=DEFAULT_METADATA,
        help="Path to metadata.json produced by download_dataset.py.",
    )
    parser.add_argument(
        "--captions_file", type=Path, default=DEFAULT_CAPTIONS,
        help="Path to save/load the captions JSON file.",
    )
    parser.add_argument(
        "--chroma_dir", type=Path, default=DEFAULT_CHROMA_DIR,
        help="Directory for ChromaDB persistent storage.",
    )
    parser.add_argument(
        "--mode", choices=["full", "caption_only", "embed_only"], default="full",
        help="Which stages to run (default: full).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process at most N images (for quick testing).",
    )
    parser.add_argument(
        "--use_lightweight_vlm", action="store_true",
        help="Use BLIP base instead of BLIP-2 (CPU-friendly, lower quality).",
    )
    parser.add_argument(
        "--use_4bit", action="store_true",
        help="Load BLIP-2 in 4-bit quantization (saves ~8 GB VRAM).",
    )
    parser.add_argument(
        "--checkpoint_every", type=int, default=50,
        help="Save caption checkpoint every N images (default: 50).",
    )
    parser.add_argument(
        "--clip_model", type=str, default=os.getenv("CLIP_MODEL_NAME", "ViT-B-32"),
        help="OpenCLIP model name.",
    )
    parser.add_argument(
        "--clip_pretrained", type=str, default=os.getenv("CLIP_PRETRAINED", "openai"),
        help="OpenCLIP pretrained weights identifier.",
    )
    parser.add_argument(
        "--text_model", type=str, default=os.getenv("TEXT_EMBED_MODEL", "BAAI/bge-small-en-v1.5"),
        help="SentenceTransformer model for caption embedding.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size for CLIP image embedding (adjust to GPU VRAM).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("\n" + "="*60)
    print("  Glance Fashion Retrieval — Part A Indexer")
    print("="*60)
    print(f"  Mode:         {args.mode}")
    print(f"  Image dir:    {args.image_dir}")
    print(f"  Captions:     {args.captions_file}")
    print(f"  ChromaDB:     {args.chroma_dir}")
    print(f"  Limit:        {args.limit or 'none'}")
    print(f"  Lightweight:  {args.use_lightweight_vlm}")
    print(f"  4-bit quant:  {args.use_4bit}")
    print("="*60 + "\n")

    # Step 1: Load image list
    image_paths = load_image_list(args.metadata_path, args.image_dir, limit=args.limit)

    if not image_paths:
        logger.error("[Indexer] No images found. Exiting.")
        sys.exit(1)

    captions: dict[str, dict] = {}

    # Step 2: Caption stage
    if args.mode in ("full", "caption_only"):
        captions = run_caption_stage(
            image_paths=image_paths,
            captions_file=args.captions_file,
            checkpoint_path=DEFAULT_CHECKPOINT,
            use_lightweight=args.use_lightweight_vlm,
            use_4bit=args.use_4bit,
            checkpoint_every=args.checkpoint_every,
        )
        print(f"\n[OK] Caption stage complete: {len(captions)} captions generated.")

    if args.mode == "caption_only":
        print("   Mode=caption_only, stopping here.")
        return

    # Step 3: Load captions if embed_only mode
    if args.mode == "embed_only":
        if not args.captions_file.exists():
            logger.error(
                f"[Indexer] embed_only mode requires {args.captions_file}. "
                "Run --mode caption_only first."
            )
            sys.exit(1)
        with open(args.captions_file) as f:
            captions = json.load(f)
        logger.info(f"[Indexer] Loaded {len(captions)} existing captions for embedding.")

    # Step 4: Embed + index
    final_stats = run_embed_index_stage(
        image_paths=image_paths,
        captions=captions,
        chroma_dir=args.chroma_dir,
        clip_model=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        text_model=args.text_model,
        batch_size=args.batch_size,
    )

    print("\n" + "="*60)
    print("  Indexing Complete")
    print("="*60)
    print(f"  CLIP vectors:  {final_stats['clip_vectors']}")
    print(f"  Text vectors:  {final_stats['text_vectors']}")
    print(f"  Storage:       {args.chroma_dir}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
