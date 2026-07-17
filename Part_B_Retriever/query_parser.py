"""
Part_B_Retriever/query_parser.py
===================================
Natural language query parser: converts free-form text queries into the same
structured schema used by the caption generator, enabling attribute-level matching.

WHY QUERY PARSING?
------------------
Without query parsing, a text like "red tie and white shirt in a formal setting"
would be embedded by BGE as a single 384-d vector. The vector similarity might
capture *some* of the semantics, but it can't leverage the metadata filter
mechanism in ChromaDB (e.g., `where={"style": "formal"}`), and it loses the
compositional color-garment binding signal that our structured captions carry.

By parsing the query into the same schema as our captions, we get:
  1. ChromaDB `where` filters: hard-filter on environment/style when strongly implied
  2. Attribute-match bonus: exact-match on clothing/color pairs adds a hard score boost
  3. Query expansion: "casual weekend city walk" → explicit clothing vocabulary
     improves BGE similarity against the indexed caption texts

PRIMARY PATH — Groq LLM (llama-3.1-8b-instant):
  - Fast (~0.3s latency), free tier at reasonable limits
  - Instruction-following sufficient for structured JSON output
  - Temperature 0.1 for deterministic output (don't want hallucinations here)
  - Few-shot examples in the prompt for reliable JSON structure

FALLBACK PATH — Rule-based parser:
  - Zero external dependencies (works 100% offline)
  - Coverage: color vocabulary list, garment vocabulary list, environment/style
    keyword dictionaries, proximity-based color-garment binding
  - Limitation: misses stylistic inference ("business casual" from contextual cues)
  - Always available as a safety net if Groq API is unavailable or rate-limited

CONSERVATIVE FILTER POLICY:
  The `filter_environment` and `filter_style` flags are intentionally conservative.
  Only set them to True when the query *explicitly* names an environment or style.
  Aggressively filtering kills recall — a query for "red tie formal setting" with
  a strict `where={"style": "formal"}` filter would miss images tagged "business_casual"
  that might visually match better. We prefer soft-scoring over hard-filtering.
"""

import json
import logging
import os
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Vocabulary lists (shared with caption_generator heuristic parser)
# Keep these in sync if you expand the vocabulary.
# ---------------------------------------------------------------------------

COLOR_VOCABULARY = [
    "red", "blue", "green", "yellow", "orange", "purple", "pink",
    "white", "black", "grey", "gray", "brown", "beige", "navy",
    "teal", "olive", "maroon", "burgundy", "lavender", "coral",
    "cream", "ivory", "khaki", "mustard", "cyan", "magenta",
    "gold", "silver", "bronze", "charcoal", "indigo", "turquoise",
    "salmon", "mint", "peach", "lilac", "tan", "camel", "bright yellow",
]

GARMENT_VOCABULARY = [
    "shirt", "t-shirt", "tee", "blouse", "top", "tank top",
    "jacket", "coat", "blazer", "cardigan", "hoodie", "sweater", "pullover",
    "dress", "skirt", "suit", "tuxedo", "jumpsuit",
    "pants", "trousers", "jeans", "shorts", "leggings", "chinos", "slacks",
    "shoes", "sneakers", "boots", "heels", "loafers", "sandals", "oxfords",
    "tie", "scarf", "hat", "cap", "beanie",
    "bag", "purse", "backpack", "belt", "watch", "glasses", "sunglasses", "gloves",
    "raincoat", "overcoat", "trench coat", "parka",
    "vest", "waistcoat", "polo",
]

ENVIRONMENT_KEYWORDS: dict[str, list[str]] = {
    "office":       ["office", "workspace", "desk", "corporate", "workplace", "business meeting", "boardroom"],
    "urban_street": ["street", "urban", "city", "sidewalk", "downtown", "city walk", "alley"],
    "park":         ["park", "outdoor", "garden", "grass", "nature", "trees", "bench", "picnic"],
    "home":         ["home", "house", "living room", "bedroom", "kitchen", "couch", "sofa", "interior"],
    "restaurant":   ["restaurant", "cafe", "dining", "table", "food", "bar", "bistro"],
    "gym":          ["gym", "fitness", "workout", "exercise", "training", "athletic"],
    "beach":        ["beach", "ocean", "sea", "sand", "water", "shore", "coastal"],
    "studio":       ["studio", "photo shoot", "backdrop", "set", "model shoot"],
}

STYLE_KEYWORDS: dict[str, list[str]] = {
    "formal":          ["formal", "black tie", "tuxedo", "gown", "suit and tie"],
    "business_casual": ["business casual", "business attire", "professional", "office wear"],
    "smart_casual":    ["smart casual", "polished casual", "neat casual"],
    "casual":          ["casual", "relaxed", "everyday", "weekend", "laid-back", "comfortable"],
    "streetwear":      ["streetwear", "street style", "hypebeast", "urban fashion", "skate"],
    "athleisure":      ["athleisure", "athletic", "sportswear", "activewear", "workout wear"],
    "evening":         ["evening", "night out", "cocktail", "gala", "party", "date night"],
}

# Few-shot examples for the Groq prompt — critical for reliable JSON output.
# We include one compositional example and one style-inference example per the spec.
_FEW_SHOT_EXAMPLES = """
Example 1 (compositional query):
Query: "A red tie and a white shirt in a formal setting."
Output:
{
  "clothing_items": ["tie", "shirt"],
  "colors": ["red tie", "white shirt"],
  "environment": "office",
  "style": "formal",
  "accessories": [],
  "expanded_query": "A man wearing a red necktie paired with a white dress shirt in a formal business or black-tie setting.",
  "filter_environment": false,
  "filter_style": true
}

Example 2 (style inference from context):
Query: "Casual weekend outfit for a city walk."
Output:
{
  "clothing_items": ["jeans", "sneakers", "t-shirt", "jacket"],
  "colors": [],
  "environment": "urban_street",
  "style": "casual",
  "accessories": [],
  "expanded_query": "A person in casual everyday clothing like jeans, sneakers, and a relaxed top or jacket, walking through an urban city environment on a weekend.",
  "filter_environment": false,
  "filter_style": false
}

Example 3 (attribute specific):
Query: "A person in a bright yellow raincoat."
Output:
{
  "clothing_items": ["raincoat"],
  "colors": ["bright yellow raincoat"],
  "environment": "unknown",
  "style": "casual",
  "accessories": [],
  "expanded_query": "A person wearing a vivid bright yellow waterproof rain jacket or raincoat.",
  "filter_environment": false,
  "filter_style": false
}
"""

_SYSTEM_PROMPT = (
    "You are a fashion image search assistant. Your job is to parse natural language "
    "search queries into structured JSON for a fashion retrieval system. "
    "Return ONLY valid JSON with these exact fields. No other text."
)

_USER_PROMPT_TEMPLATE = (
    "{few_shot}\n\n"
    "Now parse this query:\n"
    "Query: \"{query}\"\n"
    "Output:"
)


# ---------------------------------------------------------------------------
# Parsed query dataclass
# ---------------------------------------------------------------------------

class ParsedQuery:
    """
    Structured representation of a natural language query.
    Mirrors the StructuredCaption schema so attribute matching is straightforward.
    """

    def __init__(
        self,
        clothing_items: list[str] = None,
        colors: list[str] = None,
        environment: str = "unknown",
        style: str = "unknown",
        accessories: list[str] = None,
        expanded_query: str = "",
        filter_environment: bool = False,
        filter_style: bool = False,
        parse_method: str = "unknown",
    ):
        self.clothing_items = clothing_items or []
        self.colors = colors or []
        self.environment = environment
        self.style = style
        self.accessories = accessories or []
        self.expanded_query = expanded_query
        self.filter_environment = filter_environment
        self.filter_style = filter_style
        self.parse_method = parse_method   # "llm", "rule_based", or "fallback"

    def to_dict(self) -> dict:
        return {
            "clothing_items":    self.clothing_items,
            "colors":            self.colors,
            "environment":       self.environment,
            "style":             self.style,
            "accessories":       self.accessories,
            "expanded_query":    self.expanded_query,
            "filter_environment": self.filter_environment,
            "filter_style":      self.filter_style,
            "parse_method":      self.parse_method,
        }

    def build_chroma_filter(self) -> Optional[dict]:
        """
        Build a ChromaDB `where` clause from this parsed query.

        CONSERVATIVE POLICY: Only add filters when the query explicitly signals
        the attribute AND the filter_* flag is True. This prevents over-filtering
        that would kill recall on ambiguous queries.

        ChromaDB where syntax: {"field": {"$eq": "value"}} for exact match.
        Returns None if no filters are warranted (retriever should not pass a
        `where` arg to ChromaDB in this case — it's different from passing `{}`).
        """
        conditions: list[dict] = []

        if self.filter_environment and self.environment and self.environment != "unknown":
            conditions.append({"environment": {"$eq": self.environment}})

        if self.filter_style and self.style and self.style != "unknown":
            conditions.append({"style": {"$eq": self.style}})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        # ChromaDB AND: {"$and": [...]}
        return {"$and": conditions}


# ---------------------------------------------------------------------------
# LLM-based parser (primary path)
# ---------------------------------------------------------------------------

class GroqQueryParser:
    """
    Query parser using Groq's llama-3.1-8b-instant.

    Groq was chosen over OpenAI for the free tier and fast inference (~0.3s).
    llama-3.1-8b is instruction-tuned with strong JSON output capability when
    prompted with few-shot examples and a clear schema.

    Args:
        api_key:   Groq API key. If None, falls back to GROQ_API_KEY env var.
        model:     Groq model identifier.
        max_retries: Retry count on transient API errors.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llama-3.1-8b-instant",
        max_retries: int = 2,
    ):
        self.model = model
        self.max_retries = max_retries
        self._client = None

        resolved_key = api_key or os.getenv("GROQ_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Groq API key not found. Set GROQ_API_KEY env var or pass api_key= argument."
            )
        try:
            from groq import Groq
            self._client = Groq(api_key=resolved_key)
        except ImportError:
            raise ImportError("groq package not installed. Run: pip install groq")

    def parse(self, query: str) -> ParsedQuery:
        """
        Parse a query using the Groq LLM.

        On transient API errors, retries up to max_retries times.
        On persistent failure, raises so the caller can fall back to rule-based.
        """
        prompt = _USER_PROMPT_TEMPLATE.format(
            few_shot=_FEW_SHOT_EXAMPLES,
            query=query,
        )

        last_exc = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,   # low temperature = deterministic JSON output
                    max_tokens=400,
                )
                raw_text = response.choices[0].message.content.strip()
                parsed = self._parse_response(raw_text)
                parsed.parse_method = "llm"
                return parsed

            except Exception as exc:
                last_exc = exc
                logger.warning(
                    f"[GroqQueryParser] Attempt {attempt+1}/{self.max_retries+1} failed: {exc}"
                )

        raise RuntimeError(f"Groq parsing failed after {self.max_retries+1} attempts: {last_exc}")

    def _parse_response(self, text: str) -> ParsedQuery:
        """Parse LLM response into ParsedQuery, with JSON extraction fallback."""
        # Tier 1: direct parse
        try:
            data = json.loads(text)
            return self._dict_to_parsed_query(data)
        except (json.JSONDecodeError, ValueError):
            pass

        # Tier 2: regex-extract JSON block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                return self._dict_to_parsed_query(data)
            except (json.JSONDecodeError, ValueError):
                pass

        logger.warning("[GroqQueryParser] LLM output not parseable as JSON; using empty result.")
        return ParsedQuery(expanded_query=text[:200], parse_method="llm_partial")

    @staticmethod
    def _dict_to_parsed_query(data: dict) -> ParsedQuery:
        from Part_A_Indexer.caption_generator import VALID_ENVIRONMENTS, VALID_STYLES
        env = data.get("environment", "unknown")
        style = data.get("style", "unknown")
        return ParsedQuery(
            clothing_items=data.get("clothing_items", []),
            colors=data.get("colors", []),
            environment=env if env in VALID_ENVIRONMENTS else "unknown",
            style=style if style in VALID_STYLES else "unknown",
            accessories=data.get("accessories", []),
            expanded_query=data.get("expanded_query", ""),
            filter_environment=bool(data.get("filter_environment", False)),
            filter_style=bool(data.get("filter_style", False)),
        )


# ---------------------------------------------------------------------------
# Rule-based fallback parser (zero dependencies, works offline)
# ---------------------------------------------------------------------------

class RuleBasedQueryParser:
    """
    Offline rule-based query parser.

    Uses vocabulary matching + proximity-based color-garment binding.
    Quality is lower than the LLM parser (especially for style inference),
    but it's always available and deterministic.

    Binding logic: for each garment found in the query, scan a window of 4
    tokens to the left for a color word. This catches "red tie" and "white shirt"
    in "a red tie and a white shirt" without requiring LLM inference.
    """

    def parse(self, query: str) -> ParsedQuery:
        query_lower = query.lower()
        words = query_lower.split()

        # --- Color-garment proximity binding ---
        found_colors: list[str] = []
        found_garments: list[str] = []

        # Multi-word garments first (e.g., "t-shirt", "trench coat", "rain coat")
        sorted_garments = sorted(GARMENT_VOCABULARY, key=len, reverse=True)
        remaining_text = query_lower
        for garment in sorted_garments:
            if garment in remaining_text:
                found_garments.append(garment)
                # Find position and scan left for a color
                idx = remaining_text.find(garment)
                prefix_words = remaining_text[:idx].split()
                for w in reversed(prefix_words[-4:]):   # look back 4 tokens
                    clean = re.sub(r"[^\w]", "", w)
                    if clean in COLOR_VOCABULARY:
                        # Avoid double-binding same color
                        binding = f"{clean} {garment}"
                        if binding not in found_colors:
                            found_colors.append(binding)
                        break

        # --- Single-color mentions not attached to garments ---
        unbound_colors = [
            c for c in COLOR_VOCABULARY
            if c in query_lower and not any(c in fc for fc in found_colors)
        ]

        # --- Environment detection ---
        detected_env = "unknown"
        for env, kws in ENVIRONMENT_KEYWORDS.items():
            if any(kw in query_lower for kw in kws):
                detected_env = env
                break

        # --- Style detection ---
        detected_style = "unknown"
        for style, kws in STYLE_KEYWORDS.items():
            if any(kw in query_lower for kw in kws):
                detected_style = style
                break

        # --- Conservative filter flags ---
        # Only filter on environment/style if the keyword is very explicit
        filter_env = detected_env != "unknown" and any(
            kw in query_lower
            for kw in (ENVIRONMENT_KEYWORDS.get(detected_env, []))
            if len(kw) > 5   # short generic words like "home" might be accidental
        )
        filter_style = detected_style in ("formal", "business_casual") and any(
            kw in query_lower
            for kw in STYLE_KEYWORDS.get(detected_style, [])
        )

        # --- Expand query with detected vocabulary ---
        expansion_parts = [query]
        if found_garments:
            expansion_parts.append(f"Clothing: {', '.join(found_garments)}.")
        if found_colors:
            expansion_parts.append(f"Colors: {', '.join(found_colors)}.")
        if detected_env != "unknown":
            expansion_parts.append(f"Setting: {detected_env.replace('_', ' ')}.")
        if detected_style != "unknown":
            expansion_parts.append(f"Style: {detected_style.replace('_', ' ')}.")

        return ParsedQuery(
            clothing_items=list(dict.fromkeys(found_garments)),
            colors=list(dict.fromkeys(found_colors)),
            environment=detected_env,
            style=detected_style,
            accessories=[],
            expanded_query=" ".join(expansion_parts),
            filter_environment=filter_env,
            filter_style=filter_style,
            parse_method="rule_based",
        )


# ---------------------------------------------------------------------------
# Unified parser — tries LLM first, falls back to rule-based
# ---------------------------------------------------------------------------

class QueryParser:
    """
    Main query parser entry point.

    Tries the Groq LLM parser first; on failure (no API key, rate limit,
    network error), falls back to the rule-based parser transparently.

    Args:
        groq_api_key: Optional API key override. If None, reads GROQ_API_KEY env.
    """

    def __init__(self, groq_api_key: Optional[str] = None):
        self._llm_parser: Optional[GroqQueryParser] = None
        self._rule_parser = RuleBasedQueryParser()

        try:
            self._llm_parser = GroqQueryParser(api_key=groq_api_key)
            logger.info("[QueryParser] Groq LLM parser initialized.")
        except (ValueError, ImportError) as exc:
            logger.warning(
                f"[QueryParser] Groq parser unavailable ({exc}). "
                "Rule-based parser will be used for all queries."
            )

    def parse(self, query: str) -> ParsedQuery:
        """
        Parse a query, trying LLM first and falling back to rule-based.

        Args:
            query: Raw natural language search query.

        Returns:
            ParsedQuery with all fields populated.
        """
        if self._llm_parser is not None:
            try:
                result = self._llm_parser.parse(query)
                logger.debug(f"[QueryParser] LLM parsed: {result.to_dict()}")
                return result
            except Exception as exc:
                logger.warning(
                    f"[QueryParser] LLM parser failed ({exc}), falling back to rule-based."
                )

        result = self._rule_parser.parse(query)
        logger.debug(f"[QueryParser] Rule-based parsed: {result.to_dict()}")
        return result
