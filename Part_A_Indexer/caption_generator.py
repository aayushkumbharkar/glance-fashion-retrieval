"""
Part_A_Indexer/caption_generator.py
=====================================
VLM-based structured caption generation for fashion images.

WHY STRUCTURED CAPTIONS?
-------------------------
Vanilla CLIP produces a single holistic embedding per image. This embedding
conflates visual semantics without preserving *which* attribute belongs to
*which* garment — the "compositional binding failure" described in the README.
By prompting a VLM to output structured JSON (color → garment mapping, explicit
environment/style tags), we convert each image into a partially-structured
document that enables:
  1. Hard-filter queries  ("show only office environments")
  2. Attribute-match bonuses in the fusion scorer
  3. A high-quality text description for embedding by a sentence transformer
     that was trained on structured retrieval tasks (BGE), unlike CLIP's text
     encoder which was trained only to match image-level alt-text.

ARCHITECTURE CHOICE — BLIP-2 / Flan-T5-XL:
  BLIP-2 was chosen over GPT-4V / LLaVA for three reasons:
  1. It runs locally with no API costs (important for batch indexing 1000 images).
  2. Flan-T5-XL's instruction-following ability is sufficient to reliably output JSON
     when the prompt is carefully constrained (we validate this with 3-tier fallback).
  3. It has reasonable VRAM footprint (8 GB FP16 or 6 GB INT8 with bitsandbytes),
     fitting on a consumer GPU.

  Fallback to BLIP base (LightweightCaptionGenerator) is provided for CPU-only
  environments. BLIP base does *not* have instruction following, so we use a
  simpler prompt and the heuristic parser does more work.

PARSING STRATEGY — 3-TIER FALLBACK:
  Tier 1: json.loads() on the raw VLM output
  Tier 2: regex-extract the first {...} block, then json.loads()
  Tier 3: keyword heuristic parser — color/garment proximity scan
  This ensures a single malformed LLM output never crashes the indexing batch.
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema — all downstream components depend on this exact structure.
# Changing field names here requires updating vector_store.py metadata keys,
# query_parser.py output schema, and retriever.py attribute bonus logic.
# ---------------------------------------------------------------------------

VALID_ENVIRONMENTS = {
    "office", "urban_street", "park", "home",
    "restaurant", "gym", "beach", "studio", "unknown",
}

VALID_STYLES = {
    "formal", "business_casual", "smart_casual", "casual",
    "streetwear", "athleisure", "evening", "unknown",
}

CAPTION_SCHEMA: dict[str, Any] = {
    "clothing_items": [],       # e.g. ["blazer", "chinos", "sneakers"]
    "colors": [],               # e.g. ["navy blazer", "beige chinos", "white sneakers"]
    "environment": "unknown",   # one of VALID_ENVIRONMENTS
    "style": "unknown",         # one of VALID_STYLES
    "accessories": [],          # e.g. ["watch", "belt"]
    "dominant_color": "unknown",
    "vibe_description": "",     # a natural sentence the sentence embedder can embed
}

# Color name vocabulary for the heuristic parser (Tier 3).
# We keep a comprehensive list to maximize recall in the fallback path.
COLOR_VOCABULARY = [
    "red", "blue", "green", "yellow", "orange", "purple", "pink",
    "white", "black", "grey", "gray", "brown", "beige", "navy",
    "teal", "olive", "maroon", "burgundy", "lavender", "coral",
    "cream", "ivory", "khaki", "mustard", "cyan", "magenta",
    "gold", "silver", "bronze", "charcoal", "indigo", "turquoise",
    "salmon", "mint", "peach", "lilac", "tan", "camel",
]

GARMENT_VOCABULARY = [
    "shirt", "t-shirt", "tee", "blouse", "top",
    "jacket", "coat", "blazer", "cardigan", "hoodie", "sweater",
    "dress", "skirt", "suit",
    "pants", "trousers", "jeans", "shorts", "leggings",
    "shoes", "sneakers", "boots", "heels", "loafers", "sandals",
    "tie", "scarf", "hat", "cap", "bag", "purse", "backpack",
    "belt", "watch", "glasses", "sunglasses", "gloves",
    "raincoat", "overcoat", "trench",
]


# ---------------------------------------------------------------------------
# Caption dataclass — typed container so callers don't rely on raw dicts
# ---------------------------------------------------------------------------

@dataclass
class StructuredCaption:
    clothing_items: list[str] = field(default_factory=list)
    colors: list[str] = field(default_factory=list)     # "color garment" convention
    environment: str = "unknown"
    style: str = "unknown"
    accessories: list[str] = field(default_factory=list)
    dominant_color: str = "unknown"
    vibe_description: str = ""
    raw_vlm_output: str = ""          # preserved for debugging
    parse_tier_used: int = -1         # 1/2/3 tells us how well the VLM behaved

    def to_dict(self) -> dict:
        return {
            "clothing_items": self.clothing_items,
            "colors": self.colors,
            "environment": self.environment,
            "style": self.style,
            "accessories": self.accessories,
            "dominant_color": self.dominant_color,
            "vibe_description": self.vibe_description,
            "raw_vlm_output": self.raw_vlm_output,
            "parse_tier_used": self.parse_tier_used,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StructuredCaption":
        return cls(
            clothing_items=d.get("clothing_items", []),
            colors=d.get("colors", []),
            environment=d.get("environment", "unknown"),
            style=d.get("style", "unknown"),
            accessories=d.get("accessories", []),
            dominant_color=d.get("dominant_color", "unknown"),
            vibe_description=d.get("vibe_description", ""),
            raw_vlm_output=d.get("raw_vlm_output", ""),
            parse_tier_used=d.get("parse_tier_used", -1),
        )


# ---------------------------------------------------------------------------
# Helper: caption → natural prose for sentence embedding
# ---------------------------------------------------------------------------

def caption_to_text(caption: StructuredCaption) -> str:
    """
    Convert a StructuredCaption into a single prose string for embedding.

    WHY PROSE INSTEAD OF RAW JSON?
    Sentence transformers (BGE, etc.) are trained on natural language sentence pairs.
    Feeding them raw JSON produces suboptimal embeddings because the tokenizer treats
    punctuation like braces and colons as noise. Converting to readable prose
    aligns the input distribution with what the model was trained on.

    We intentionally concatenate multiple fields in a consistent order so the
    sentence embedding captures all structured signals.
    """
    parts: list[str] = []

    if caption.colors:
        # "navy blazer, beige chinos" — the color-garment binding is the key
        # compositional signal we want preserved in the text embedding space.
        parts.append("The person is wearing " + ", ".join(caption.colors) + ".")

    if caption.clothing_items:
        remaining = [
            item for item in caption.clothing_items
            if not any(item.lower() in c.lower() for c in caption.colors)
        ]
        if remaining:
            parts.append("Clothing items include " + ", ".join(remaining) + ".")

    if caption.accessories:
        parts.append("Accessories: " + ", ".join(caption.accessories) + ".")

    if caption.environment and caption.environment != "unknown":
        env_map = {
            "office": "They are in an office setting.",
            "urban_street": "The setting is an urban street.",
            "park": "The photo is taken in a park.",
            "home": "The person is at home.",
            "restaurant": "The setting is a restaurant.",
            "gym": "They appear to be at the gym.",
            "beach": "The background is a beach.",
            "studio": "This looks like a studio or photo shoot.",
        }
        parts.append(env_map.get(caption.environment, f"Environment: {caption.environment}."))

    if caption.style and caption.style != "unknown":
        style_map = {
            "formal": "The overall style is formal.",
            "business_casual": "The outfit is business casual.",
            "smart_casual": "The look is smart casual.",
            "casual": "The style is casual and relaxed.",
            "streetwear": "The outfit has a streetwear aesthetic.",
            "athleisure": "The style is athleisure.",
            "evening": "This is an evening or formal event outfit.",
        }
        parts.append(style_map.get(caption.style, f"Style: {caption.style}."))

    if caption.vibe_description:
        parts.append(caption.vibe_description)

    return " ".join(parts) if parts else "A person wearing clothing."


# ---------------------------------------------------------------------------
# 3-tier JSON parser
# ---------------------------------------------------------------------------

def _normalize_caption_dict(raw: dict) -> dict:
    """
    Enforce schema consistency on a parsed caption dict.
    Handles field name mismatches (e.g. 'color_items' vs 'colors'),
    invalid enum values, and missing fields.
    """
    # Alias tolerance for common VLM mistakes
    aliases = {
        "garments": "clothing_items",
        "items": "clothing_items",
        "clothing": "clothing_items",
        "color_items": "colors",
        "color_garment": "colors",
        "setting": "environment",
        "location": "environment",
        "outfit_style": "style",
        "outfit": "style",
        "dominant": "dominant_color",
        "main_color": "dominant_color",
        "vibe": "vibe_description",
        "description": "vibe_description",
    }
    normalized: dict = {}
    for k, v in raw.items():
        canonical = aliases.get(k.lower(), k.lower())
        normalized[canonical] = v

    result = dict(CAPTION_SCHEMA)   # start from defaults
    result.update(normalized)

    # Enforce list types
    for list_field in ("clothing_items", "colors", "accessories"):
        if isinstance(result[list_field], str):
            result[list_field] = [result[list_field]] if result[list_field] else []
        elif not isinstance(result[list_field], list):
            result[list_field] = []

    # Enforce enum validity — degrade to "unknown" rather than propagating garbage
    if result["environment"] not in VALID_ENVIRONMENTS:
        result["environment"] = "unknown"
    if result["style"] not in VALID_STYLES:
        result["style"] = "unknown"

    return result


def _tier1_parse(text: str) -> Optional[dict]:
    """Tier 1: Direct json.loads — works when the VLM returns clean JSON."""
    try:
        data = json.loads(text.strip())
        if isinstance(data, dict):
            return _normalize_caption_dict(data)
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def _tier2_parse(text: str) -> Optional[dict]:
    """
    Tier 2: Regex-extract the first {...} block in the text, then parse.
    Handles VLMs that prefix their JSON output with a sentence like
    "Here is the JSON:" or wrap it in markdown code fences.
    """
    # Match from first { to last } — greedy to capture nested structures
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict):
                return _normalize_caption_dict(data)
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def _tier3_parse(text: str, image_path: Optional[str] = None) -> dict:
    """
    Tier 3: Keyword heuristic parser.

    When the VLM output is completely unparseable as JSON, we fall back to
    scanning the raw text for known color and garment keywords, using word
    proximity to infer color-garment bindings.

    This is intentionally conservative: it produces a partial caption with
    high precision (the attributes we do find are usually right) but lower
    recall than a good VLM response. The structured text embedding will still
    carry partial signal.

    Environment and style are inferred from keyword sets.
    """
    text_lower = text.lower()
    result = dict(CAPTION_SCHEMA)

    # --- Color-garment proximity binding ---
    # For each garment found, scan a window of 4 tokens to the left for a color.
    words = text_lower.split()
    found_colors: list[str] = []
    found_garments: list[str] = []

    for i, word in enumerate(words):
        # Strip punctuation from word for matching
        clean = re.sub(r"[^\w]", "", word)
        if clean in GARMENT_VOCABULARY:
            found_garments.append(clean)
            # Look left up to 4 tokens for a color
            window = words[max(0, i - 4):i]
            for w in reversed(window):
                clean_w = re.sub(r"[^\w]", "", w)
                if clean_w in COLOR_VOCABULARY:
                    found_colors.append(f"{clean_w} {clean}")
                    break

    result["clothing_items"] = list(dict.fromkeys(found_garments))[:6]   # dedup, cap at 6
    result["colors"] = list(dict.fromkeys(found_colors))[:6]

    # --- Dominant color: most frequent color in text ---
    color_counts = {c: text_lower.count(c) for c in COLOR_VOCABULARY}
    dominant = max(color_counts, key=color_counts.get) if any(color_counts.values()) else "unknown"
    result["dominant_color"] = dominant

    # --- Environment keyword matching ---
    env_keywords = {
        "office": ["office", "desk", "corporate", "workplace", "cubicle"],
        "urban_street": ["street", "urban", "city", "sidewalk", "downtown"],
        "park": ["park", "outdoor", "garden", "grass", "nature", "trees"],
        "home": ["home", "house", "living room", "bedroom", "kitchen", "couch"],
        "restaurant": ["restaurant", "cafe", "dining", "table", "food"],
        "gym": ["gym", "fitness", "workout", "exercise", "training"],
        "beach": ["beach", "ocean", "sea", "sand", "water", "shore"],
        "studio": ["studio", "photo", "backdrop", "shoot", "model"],
    }
    for env, kws in env_keywords.items():
        if any(kw in text_lower for kw in kws):
            result["environment"] = env
            break

    # --- Style keyword matching ---
    style_keywords = {
        "formal": ["formal", "suit", "tuxedo", "gown", "black tie"],
        "business_casual": ["business", "office", "professional", "blazer"],
        "smart_casual": ["smart casual", "neat", "polished casual"],
        "casual": ["casual", "relaxed", "everyday", "weekend"],
        "streetwear": ["streetwear", "street", "hypebeast", "urban"],
        "athleisure": ["athleisure", "athletic", "sportswear", "activewear"],
        "evening": ["evening", "night out", "cocktail", "gala", "party"],
    }
    for style, kws in style_keywords.items():
        if any(kw in text_lower for kw in kws):
            result["style"] = style
            break

    # Vibe description: use the raw text as-is, truncated — gives the sentence
    # embedder something to work with even though it's not structured.
    result["vibe_description"] = text[:300] if text else ""

    return result


def parse_vlm_output(raw_text: str) -> tuple[dict, int]:
    """
    Attempt to parse VLM output through all three tiers.

    Returns:
        (caption_dict, tier_used) where tier_used is 1, 2, or 3.
    """
    result = _tier1_parse(raw_text)
    if result is not None:
        return result, 1

    result = _tier2_parse(raw_text)
    if result is not None:
        return result, 2

    logger.warning("VLM output not parseable as JSON (tiers 1 & 2 failed). Using heuristic parser.")
    return _tier3_parse(raw_text), 3


# ---------------------------------------------------------------------------
# BLIP-2 caption generator — primary path (GPU recommended)
# ---------------------------------------------------------------------------

class CaptionGenerator:
    """
    Structured caption generator using Salesforce/blip2-flan-t5-xl.

    BLIP-2 architecture:
      - A frozen visual encoder (ViT-g) extracts image features
      - A Querying Transformer (Q-Former) bridges vision and language
      - Flan-T5-XL (instruction-tuned T5) generates text conditioned on queries
    
    The instruction-following capability of Flan-T5-XL is what allows us to
    prompt for structured JSON output. BLIP-1 / base BLIP can't reliably follow
    structured output instructions, which is why BLIP-2 was chosen as the
    primary path.

    4-bit quantization (bitsandbytes):
      Loading BLIP-2 FP16 requires ~14 GB VRAM. With INT8 quantization it drops
      to ~8 GB, and with 4-bit (NF4) it drops to ~6 GB. We support `use_4bit`
      for constrained setups.

    Args:
        model_name:    HuggingFace model identifier.
        device:        'cuda', 'cpu', or 'auto' (let accelerate decide).
        use_4bit:      Enable bitsandbytes 4-bit quantization.
        checkpoint_path: Where to persist caption checkpoints.
    """

    # The structured prompt for BLIP-2 / Flan-T5. Note the explicit JSON template —
    # without this, the model tends to produce free-form text. We use a *compact*
    # template (no optional fields left empty) to minimize output tokens and reduce
    # hallucination from long generations.
    _PROMPT_TEMPLATE = (
        'Question: Analyze this fashion image and return ONLY a JSON object with these exact fields: '
        '"clothing_items" (list of garment names), '
        '"colors" (list of strings in "color garment" format, e.g. "navy blazer"), '
        '"environment" (one of: office, urban_street, park, home, restaurant, gym, beach, studio, unknown), '
        '"style" (one of: formal, business_casual, smart_casual, casual, streetwear, athleisure, evening, unknown), '
        '"accessories" (list of accessories), '
        '"dominant_color" (most prominent color as a single word), '
        '"vibe_description" (one sentence describing the overall look). '
        'Return ONLY valid JSON, no other text. Answer:'
    )

    def __init__(
        self,
        model_name: str = "Salesforce/blip2-flan-t5-xl",
        device: str = "auto",
        use_4bit: bool = False,
        checkpoint_path: Optional[Path] = None,
    ):
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path

        # Deferred imports — we don't want transformers to load at module import
        # time because it's slow and tests don't always need it.
        from transformers import Blip2ForConditionalGeneration, Blip2Processor

        self.processor = Blip2Processor.from_pretrained(model_name)

        load_kwargs: dict = {}
        if use_4bit:
            try:
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",           # NF4 is better than INT4 for weights
                    bnb_4bit_compute_dtype=torch.float16,
                )
                load_kwargs["quantization_config"] = bnb_config
                load_kwargs["device_map"] = "auto"       # accelerate handles placement
                logger.info("[CaptionGenerator] Loading BLIP-2 with 4-bit quantization.")
            except ImportError:
                logger.warning("bitsandbytes not available; loading in FP16 instead of 4-bit.")
                load_kwargs["torch_dtype"] = torch.float16
                load_kwargs["device_map"] = "auto"
        elif device == "auto":
            load_kwargs["torch_dtype"] = torch.float16
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["torch_dtype"] = torch.float16

        self.model = Blip2ForConditionalGeneration.from_pretrained(model_name, **load_kwargs)

        if device != "auto" and not use_4bit:
            self.model = self.model.to(device)

        self.device = device
        self._checkpoint: dict[str, dict] = {}
        if checkpoint_path and checkpoint_path.exists():
            with open(checkpoint_path, "r") as f:
                self._checkpoint = json.load(f)
            logger.info(f"[CaptionGenerator] Loaded {len(self._checkpoint)} from checkpoint.")

    def generate_caption(self, image_path: Path) -> StructuredCaption:
        """
        Generate a structured caption for a single image.

        Args:
            image_path: Path to the image file.

        Returns:
            StructuredCaption with all fields populated (falling back on failure).
        """
        image_id = image_path.stem

        # Resume: skip if already processed
        if image_id in self._checkpoint:
            return StructuredCaption.from_dict(self._checkpoint[image_id])

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            logger.warning(f"[CaptionGenerator] Failed to open {image_path}: {exc}")
            return self._make_empty_caption()

        try:
            inputs = self.processor(
                images=image,
                text=self._PROMPT_TEMPLATE,
                return_tensors="pt",
            )
            # Move inputs to the same device as model
            if self.device != "auto":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=300,   # enough for the JSON structure
                    min_new_tokens=50,    # prevent premature EOS before closing brace
                    num_beams=4,          # beam search: better JSON coherence than greedy
                    early_stopping=True,
                )
            raw_text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]

        except Exception as exc:
            logger.warning(f"[CaptionGenerator] VLM inference failed for {image_path}: {exc}")
            return self._make_empty_caption()

        caption_dict, tier = parse_vlm_output(raw_text)
        caption = StructuredCaption(
            **{k: caption_dict[k] for k in CAPTION_SCHEMA.keys()},
            raw_vlm_output=raw_text,
            parse_tier_used=tier,
        )
        self._checkpoint[image_id] = caption.to_dict()
        return caption

    def generate_batch(
        self,
        image_paths: list[Path],
        checkpoint_every: int = 50,
    ) -> dict[str, StructuredCaption]:
        """
        Generate captions for a batch of images with periodic checkpointing.

        Checkpoints every `checkpoint_every` images so a crash loses at most
        that many captions. The checkpoint is appended (not rewritten) on each
        save to minimize I/O.

        Args:
            image_paths:       List of image paths to process.
            checkpoint_every:  How often to flush checkpoint to disk.

        Returns:
            Dict mapping image_id → StructuredCaption.
        """
        results: dict[str, StructuredCaption] = {}
        newly_processed = 0

        for idx, image_path in enumerate(image_paths):
            image_id = image_path.stem
            caption = self.generate_caption(image_path)
            results[image_id] = caption
            newly_processed += 1

            # Checkpoint periodically
            if self.checkpoint_path and (newly_processed % checkpoint_every == 0):
                self._save_checkpoint()
                logger.info(
                    f"[CaptionGenerator] Checkpoint @ {idx+1}/{len(image_paths)} images."
                )

        # Final checkpoint flush
        if self.checkpoint_path:
            self._save_checkpoint()

        return results

    def _save_checkpoint(self) -> None:
        """Atomically write the checkpoint dict to disk."""
        if not self.checkpoint_path:
            return
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.checkpoint_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(self._checkpoint, f)
        tmp.replace(self.checkpoint_path)

    @staticmethod
    def _make_empty_caption() -> StructuredCaption:
        """Return a default empty caption on unrecoverable errors."""
        return StructuredCaption(parse_tier_used=-1)


# ---------------------------------------------------------------------------
# Lightweight fallback — BLIP base for CPU-only environments
# ---------------------------------------------------------------------------

class LightweightCaptionGenerator:
    """
    Caption generator using Salesforce/blip-image-captioning-base.

    WHY THIS EXISTS:
    BLIP-2 / Flan-T5-XL requires ~6–14 GB VRAM depending on quantization.
    On CPU-only machines this would take hours per image. BLIP base uses a
    much smaller BERT-based decoder and runs in seconds per image on CPU.

    TRADEOFF:
    BLIP base does NOT follow structured-output instructions reliably. It
    produces a free-form caption like "a woman wearing a red dress in a park."
    We run this through the Tier-3 heuristic parser, which extracts color-garment
    pairs via proximity scanning. The resulting structured caption has lower quality
    than the BLIP-2 path (fewer accessories captured, vibe_description is just
    the raw caption) but is still useful for retrieval.

    Args:
        model_name:       HuggingFace identifier. Default is BLIP base.
        checkpoint_path:  Persistence path (same interface as CaptionGenerator).
    """

    _BLIP_BASE_MODEL = "Salesforce/blip-image-captioning-base"

    def __init__(
        self,
        model_name: str = _BLIP_BASE_MODEL,
        checkpoint_path: Optional[Path] = None,
    ):
        from transformers import BlipForConditionalGeneration, BlipProcessor

        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.checkpoint_path = checkpoint_path
        self._checkpoint: dict[str, dict] = {}

        if checkpoint_path and checkpoint_path.exists():
            with open(checkpoint_path, "r") as f:
                self._checkpoint = json.load(f)
            logger.info(f"[LightweightCaptionGenerator] Loaded {len(self._checkpoint)} from checkpoint.")

    def generate_caption(self, image_path: Path) -> StructuredCaption:
        image_id = image_path.stem
        if image_id in self._checkpoint:
            return StructuredCaption.from_dict(self._checkpoint[image_id])

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            logger.warning(f"[LightweightCaptionGenerator] Cannot open {image_path}: {exc}")
            return StructuredCaption(parse_tier_used=-1)

        try:
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=80)
            raw_text = self.processor.decode(out[0], skip_special_tokens=True)
        except Exception as exc:
            logger.warning(f"[LightweightCaptionGenerator] Inference failed for {image_path}: {exc}")
            return StructuredCaption(parse_tier_used=-1)

        # BLIP base output is free-form; apply Tier-3 heuristic directly.
        caption_dict = _tier3_parse(raw_text)
        caption_dict["vibe_description"] = raw_text   # use the BLIP caption as vibe text
        caption = StructuredCaption(
            **{k: caption_dict[k] for k in CAPTION_SCHEMA.keys()},
            raw_vlm_output=raw_text,
            parse_tier_used=3,
        )
        self._checkpoint[image_id] = caption.to_dict()
        return caption

    def generate_batch(
        self, image_paths: list[Path], checkpoint_every: int = 50
    ) -> dict[str, StructuredCaption]:
        results: dict[str, StructuredCaption] = {}
        for idx, path in enumerate(image_paths):
            results[path.stem] = self.generate_caption(path)
            if self.checkpoint_path and ((idx + 1) % checkpoint_every == 0):
                self._save_checkpoint()
        if self.checkpoint_path:
            self._save_checkpoint()
        return results

    def _save_checkpoint(self) -> None:
        if not self.checkpoint_path:
            return
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.checkpoint_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(self._checkpoint, f)
        tmp.replace(self.checkpoint_path)
