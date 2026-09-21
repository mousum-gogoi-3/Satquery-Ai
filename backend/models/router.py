import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, List, Dict, Any

from backend.utils.logger import get_logger

logger = get_logger("SatQueryAI.Router")


class QueryType(str, Enum):
    """Canonical classification categories for remote-sensing queries."""
    IMAGE_UNDERSTANDING = "image_understanding"
    GROUNDING = "grounding"
    COUNTING = "counting"
    TEMPORAL_CHANGE = "temporal_change"


@dataclass
class QueryRoutingResult:
    query_type: QueryType
    target_concept: Optional[str]
    original_query: str
    confidence: float = 1.0
    target_image_index: int = 1  # 1 for Before/First image, 2 for After/Second image, 0 for both (temporal)


class SatQueryRouter:
    """
    Intelligent Query Classifier & Router for SatQuery AI.
    Dispatches natural-language user queries to the appropriate specialized pipeline:
      1. General Image Understanding (Captioning / Scene VQA) - DEFAULT
      2. Object Location / Spatial Grounding (OWLv2 + Fusion)
      3. Object Detection & Counting
      4. Bi-Temporal Change Detection
    """

    # Target concept keyword patterns mapped to canonical labels
    CONCEPT_PATTERNS: List[Tuple[str, str]] = [
        # (regex pattern, canonical concept label)
        (r"\b(buildings?|houses?|homes?|roofs?|residential|structures?|architecture|facilities|facility)\b", "building"),
        (r"\b(roads?|highways?|streets?|roadways?|freeways?|paved(\s+roads?)?|lanes?|avenues?)\b", "road"),
        (r"\b(waters?|water\s+bodies|water\s+body|rivers?|lakes?|ponds?|reservoirs?|seas?|oceans?|canals?|creeks?|streams?)\b", "water"),
        (r"\b(agricultur(e|al)|farmlands?|farms?|farming|croplands?|crops?|cultivated|cultivation|padd(y|ies)|orchards?|pastures?|agricultural\s+fields?|crop\s+fields?)\b", "agriculture"),
        (r"\b(forests?|woodlands?|trees?|dense\s+vegetation|vegetation|jungles?|canop(y|ies))\b", "vegetation"),
        (r"\b(vehicles?|cars?|trucks?|automobiles?|buses?)\b", "vehicle"),
        (r"\b(airports?|runways?|airfields?|hangars?|airplanes?|aerodromes?)\b", "airport"),
        (r"\b(ships?|boats?|vessels?|cargos?|marines?|ferr(y|ies))\b", "ship"),
        (r"\b(bare\s+(land|soil|ground)|barren|exposed\s+soil|dirt\s+ground|unvegetated)\b", "bare_soil"),
        (r"\b(urban|cit(y|ies)|settlements?|built-?up)\b", "urban"),
        (r"\b(constructions?|excavations?|earthworks?|building\s+site)\b", "construction"),
    ]

    # Change detection intent keywords
    CHANGE_PATTERNS: List[str] = [
        r"\bwhat\s+(has\s+)?changed\b",
        r"\bchanges?\s+between\b",
        r"\bwhat\s+changed\s+between\b",
        r"\bcompare(\s+the|\s+these)?\s+(images?|rasters?|scenes?|observations?)\b",
        r"\bcompare\b",
        r"\bbefore\s+and\s+after\b",
        r"\btemporal\s+change\b",
        r"\bvegetation\s+(loss|decrease|gain|increase)\b",
        r"\bwhere\s+did\s+vegetation\s+(loss|decrease|gain|increase)\b",
        r"\b(deforest|deforestation)\b",
        r"\bwater\s+(expansion|reduction|shrink|increase|decrease)\b",
        r"\bflooding\b",
        r"\b(where\s+are\s+the\s+)?new\s+(buildings?|constructions?|structures?|roads?)\b",
        r"\b(did|has|where\s+did)\s+(any\s+)?(change|construction|vegetation\s+decrease|loss)\s+occur(red)?\b",
        r"\bconstruction\s+occur(red)?\b",
        r"\bshow(\s+me)?\s+the\s+new\s+buildings\b",
        r"\b(what|which)\s+areas?\s+changed\b",
        r"\bshow(\s+me)?\s+(the\s+)?areas?\s+that\s+changed\b",
        r"\bshow(\s+me)?\s+(the\s+)?differences?\b",
        r"\bdifferences?\s+between\b",
    ]

    # Counting intent indicators
    COUNT_PATTERNS: List[str] = [
        r"\bhow\s+many\b",
        r"^\s*count(\s+the|\s+all)?\b",
        r"\bnumber\s+of\b",
        r"\btotal\s+count\b",
        r"\bamount\s+of\b",
    ]

    # Explicit spatial grounding / location intent indicators
    GROUNDING_INDICATORS: List[str] = [
        r"\bwhere\s+(is|are|can\s+i\s+find)\b",
        r"\blocate(\s+the|\s+all)?\b",
        r"\bshow(\s+me)?(\s+the|\s+all)?\s+(location\s+of\s+|locations?\s+of\s+)?",
        r"\bfind(\s+the|\s+all)?\b",
        r"\bdetect(\s+the|\s+all)?\b",
        r"\bidentify(\s+the\s+location\s+of|\s+where)\b",
        r"\bpoint\s+out\b",
        r"\bmark(\s+the)?\b",
        r"\bspot(\s+the)?\b",
    ]

    # Explicit general understanding keywords that must NEVER trigger grounding
    EXPLICIT_UNDERSTANDING_PATTERNS: List[str] = [
        r"\bexplain(\s+this|\s+the|\s+a)?(\s+first|\s+second|\s+before|\s+after)?(\s+image|\s+scene|\s+satellite|\s+raster)?\b",
        r"\bdescribe(\s+this|\s+the|\s+a)?(\s+first|\s+second|\s+before|\s+after)?(\s+image|\s+scene|\s+satellite|\s+raster)?\b",
        r"\bwhat\s+is\s+(in\s+this|on\s+this|shown\s+in|this)\b",
        r"\bwhat\s+does\s+this\s+(image|scene|satellite)\s+show\b",
        r"\bgive\s+(me\s+)?an?\s+(overview|summary)\b",
        r"\bwhat\s+can\s+you\s+see\b",
        r"\banalyze(\s+this|\s+the)?(\s+first|\s+second|\s+before|\s+after)?(\s+image|\s+scene|\s+satellite|\s+raster)?\b",
        r"\bsummarize(\s+this|\s+the)?\b",
        r"\btell\s+me\s+about(\s+this)?\b",
        r"\boverview\s+of\s+(this|the\s+first|the\s+second|the\s+before|the\s+after)?\b",
    ]

    @classmethod
    def detect_target_image_index(cls, query: str) -> int:
        """
        Determines whether a query is addressing Image 1 (Before) or Image 2 (After).
        Returns 2 if query refers to second/after image, otherwise returns 1 (default).
        """
        q = query.lower()
        if re.search(r"\b(second|after|later|image\s*2|image\s*b)\b", q):
            return 2
        return 1

    @classmethod
    def extract_target_concept(cls, query: str) -> Optional[str]:
        """
        Extracts the canonical target object concept from a query string.
        Returns canonical label (e.g. 'building', 'road', 'water') or None if not found.
        """
        q_lower = query.lower().strip()
        for pattern, label in cls.CONCEPT_PATTERNS:
            if re.search(pattern, q_lower):
                return label
        return None

    @classmethod
    def classify_query(
        cls,
        query: str,
        has_second_image: bool = False
    ) -> QueryRoutingResult:
        """
        Classifies user query and extracts target concept.
        
        Routing Priority:
          1. Explicit Image Understanding (Overrides all else for 'explain this image', 'describe the first image', etc.)
          2. Temporal Change Analysis (If temporal intent, or 2 images + comparison/change query)
          3. Counting Questions
          4. Spatial Grounding (Only if explicit location intent + recognizable target concept)
          5. General Image Understanding (DEFAULT fallback)
        """
        q = query.lower().strip()
        target_image_idx = cls.detect_target_image_index(q)

        # Step 1: Check for explicit general understanding intent first!
        # If user asks "Explain this image", "Describe the first image", etc., it MUST go to image understanding.
        for pattern in cls.EXPLICIT_UNDERSTANDING_PATTERNS:
            if re.search(pattern, q):
                logger.info(
                    f"[SatQuery Router] Query: '{query}' | Type: image_understanding (Explicit explanation pattern) | "
                    f"Target Image: {target_image_idx} | Model: CaptionEngine / VQAEngine"
                )
                return QueryRoutingResult(
                    query_type=QueryType.IMAGE_UNDERSTANDING,
                    target_concept=None,
                    original_query=query,
                    confidence=0.98,
                    target_image_index=target_image_idx
                )

        # Step 2: Temporal Change Analysis
        # Triggered if query matches temporal patterns OR (two images present AND change/difference query)
        is_change = any(re.search(p, q) for p in cls.CHANGE_PATTERNS) or (
            has_second_image and any(w in q for w in ["change", "diff", "differ", "compare", "versus", "vs"])
        )
        if is_change:
            target = cls.extract_target_concept(q)
            logger.info(f"[SatQuery Router] Query: '{query}' | Type: temporal_change | Target: {target} | Model: BiTemporalAnalysisEngine")
            return QueryRoutingResult(
                query_type=QueryType.TEMPORAL_CHANGE,
                target_concept=target,
                original_query=query,
                confidence=0.95,
                target_image_index=0
            )

        # Step 3: Counting Questions
        is_count = any(re.search(p, q) for p in cls.COUNT_PATTERNS)
        if is_count:
            target = cls.extract_target_concept(q)
            if not target:
                cleaned = re.sub(r"how\s+many\s+", "", q).strip()
                words = cleaned.split()
                if words:
                    target = words[0].rstrip("s?,.")
                    if len(target) < 3:
                        target = "target object"
            logger.info(
                f"[SatQuery Router] Query: '{query}' | Type: counting | Target: {target} | "
                f"Target Image: {target_image_idx} | Model: OWLv2 + Fusion"
            )
            return QueryRoutingResult(
                query_type=QueryType.COUNTING,
                target_concept=target or "target object",
                original_query=query,
                confidence=0.92,
                target_image_index=target_image_idx
            )

        # Step 4: Spatial Grounding
        # Must have BOTH explicit location intent (e.g. "where is", "locate", "show me") AND a target concept
        has_grounding_intent = any(re.search(p, q) for p in cls.GROUNDING_INDICATORS)
        target = cls.extract_target_concept(q)

        if has_grounding_intent and target:
            logger.info(
                f"[SatQuery Router] Query: '{query}' | Type: grounding | Target: {target} | "
                f"Target Image: {target_image_idx} | Model: OWLv2 + Fusion"
            )
            return QueryRoutingResult(
                query_type=QueryType.GROUNDING,
                target_concept=target,
                original_query=query,
                confidence=0.95,
                target_image_index=target_image_idx
            )

        # Step 5: DEFAULT -> General Image Understanding
        # The default behavior is NEVER grounding. Any general or unclassified question uses VQA/Captioning.
        logger.info(
            f"[SatQuery Router] Query: '{query}' | Type: image_understanding (Default Fallback) | "
            f"Target Image: {target_image_idx} | Model: CaptionEngine / VQAEngine"
        )
        return QueryRoutingResult(
            query_type=QueryType.IMAGE_UNDERSTANDING,
            target_concept=target,
            original_query=query,
            confidence=0.90,
            target_image_index=target_image_idx
        )
