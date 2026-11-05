"""
Stress Vulnerability Index (SVI) Engine.

Implements Sigmoid Late Fusion:
SVI(t) = sigmoid(w_a * A(t) + w_t * T(t) + w_b * B(t) + delta_critical) * 100
where:
sigma(x) = 1 / (1 + exp(-10 * (x - 0.5)))
Weights: w_a = 0.35, w_t = 0.40, w_b = 0.25

Deterministic Hard Overrides (delta_critical):
If P_suicide > 0.80, P_panic > 0.80, or explicit markers of rape, active weapons,
or murder are detected, force delta_critical = 0.50, elevating score to CRITICAL_RISK (81-100).

EMA Smoothing:
SVI_smooth(t) = lambda * SVI(t) + (1 - lambda) * SVI_smooth(t-1) with lambda = 0.3.
"""
import math
import logging
import re
from typing import Optional, Tuple, Dict, Any

logger = logging.getLogger(__name__)

WEIGHTS = {
    "acoustic": 0.35,
    "text": 0.40,
    "behavior": 0.25,
}

# Regex and lexicon markers for emergency triggers
CRITICAL_EXPLICIT_MARKERS = [
    r"\brape\b", r"\bweapon\b", r"\bgun\b", r"\bknife\b", r"\bmurder\b",
    r"\bkilling\b", r"\bhostage\b", r"\blynch\b", r"\bmob attack\b",
    r"\bbroke our (gate|door)\b", r"\bkill myself\b", r"\bend my life\b"
]

CIVIL_PROTECTION_LEXICONS = [
    r"\blandlord\b", r"\bwell\b", r"\bdraw water\b", r"\bfield\b",
    r"\bthreatened to burn\b", r"\bfront of the whole street\b", r"\bnot let us\b",
    r"\bthreatened\b", r"\beviction\b", r"\bextortion\b", r"\bblackmail\b", r"\bharassment\b"
]


class TextClassifierPipeline:
    """
    Manages transformer-based text scoring on GPU (device='cuda') with fallback.
    - gohjiayi/suicidal-bert for P_suicide
    - IndicBERT/trauma classifier or zero-shot head for P_panic and coercion/threat violations
    """
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.suicide_model = None
        self.suicide_tokenizer = None
        self.torch_device = None
        self._init_models()

    def _init_models(self):
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            if self.device == "cuda" and torch.cuda.is_available():
                self.torch_device = torch.device("cuda")
                logger.info("Using CUDA for TextClassifierPipeline.")
            else:
                self.torch_device = torch.device("cpu")
                logger.info("Using CPU for TextClassifierPipeline.")

            # Attempt loading suicidal-bert
            try:
                model_name = "gohjiayi/suicidal-bert"
                self.suicide_tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.suicide_model = AutoModelForSequenceClassification.from_pretrained(model_name)
                self.suicide_model.to(self.torch_device)
                self.suicide_model.eval()
                logger.info(f"Loaded {model_name} onto {self.torch_device}")
            except Exception as e:
                logger.warning(f"Could not load suicidal-bert ({e}); using heuristic fallback.")
                self.suicide_model = None

        except Exception as e:
            logger.warning(f"PyTorch or transformers not available: {e}")
            self.suicide_model = None

    def analyze_text(self, text: str) -> Tuple[float, float, Dict[str, Any]]:
        """
        Returns (p_suicide, p_panic, flags_dict)
        """
        text_lower = text.lower().strip()
        p_suicide = 0.01
        p_panic = 0.05
        flags = {
            "critical_marker": False,
            "civil_rights_violation": False,
            "sc_st_violation": False,
            "primary_trigger": "Baseline Monitored Call",
        }

        if not text_lower:
            return p_suicide, p_panic, flags

        # 1. Check explicit critical markers
        for pattern in CRITICAL_EXPLICIT_MARKERS:
            if re.search(pattern, text_lower):
                flags["critical_marker"] = True
                flags["primary_trigger"] = "Active Attack & Violence Threat"
                p_panic = max(p_panic, 0.92)

        # 2. Check coercion & rights protection markers
        for pattern in CIVIL_PROTECTION_LEXICONS:
            if re.search(pattern, text_lower):
                flags["civil_rights_violation"] = True
                flags["sc_st_violation"] = True
                if not flags["critical_marker"]:
                    flags["primary_trigger"] = "Severe Coercion & Rights Deprivation"
                p_panic = max(p_panic, 0.65)

        # 3. Model inference for suicide if model is loaded
        if self.suicide_model is not None and self.suicide_tokenizer is not None:
            try:
                import torch
                inputs = self.suicide_tokenizer(
                    text, return_tensors="pt", truncation=True, max_length=128
                ).to(self.torch_device)
                with torch.no_grad():
                    logits = self.suicide_model(**inputs).logits
                    probs = torch.softmax(logits, dim=-1).squeeze().cpu().tolist()
                    # Label 1 is typically suicide/risk
                    if isinstance(probs, list) and len(probs) > 1:
                        p_suicide = float(probs[1])
                    else:
                        p_suicide = float(probs)
            except Exception as e:
                logger.error(f"Inference error in suicidal-bert: {e}")
        else:
            # Lexicon heuristic fallback for p_suicide
            suicide_keywords = ["suicide", "kill myself", "want to die", "end my life", "end it all"]
            if any(k in text_lower for k in suicide_keywords):
                p_suicide = 0.88
                flags["critical_marker"] = True
                flags["primary_trigger"] = "Explicit Suicidal Distress"

        # Panic heuristics
        panic_keywords = ["scared", "help", "hurry", "outside", "they won't let", "broke", "please", "tap once", "hiding"]
        matches = sum(1 for k in panic_keywords if k in text_lower)
        if matches > 0:
            p_panic = min(0.95, max(p_panic, 0.45 + 0.15 * matches))

        return float(p_suicide), float(p_panic), flags


# Global pipeline instance (lazy loaded)
_TEXT_CLASSIFIER: Optional[TextClassifierPipeline] = None


def get_text_classifier() -> TextClassifierPipeline:
    global _TEXT_CLASSIFIER
    if _TEXT_CLASSIFIER is None:
        _TEXT_CLASSIFIER = TextClassifierPipeline()
    return _TEXT_CLASSIFIER


def sigmoid_late_fusion(x: float) -> float:
    """
    sigma(x) = 1 / (1 + exp(-10 * (x - 0.5)))
    Scales to 0-1 range with steepness centered at x=0.5.
    """
    val = -10.0 * (x - 0.5)
    # Clamp to prevent overflow
    val = max(-30.0, min(30.0, val))
    return 1.0 / (1.0 + math.exp(val))


def score_behavior(silence_seconds: float = 0.0, is_freeze: bool = False) -> float:
    """
    B(t) in [0.0, 1.0]: Cognitive freezing / abnormal silence (>3-5s).
    """
    base = min(silence_seconds / 6.0, 1.0) * 0.70
    if is_freeze:
        base += 0.30
    return float(min(1.0, max(0.0, base)))


def compute_svi(
    a_t: float,
    t_t: float,
    b_t: float,
    p_suicide: float = 0.0,
    p_panic: float = 0.0,
    has_explicit_marker: bool = False,
) -> Tuple[float, float, str, str]:
    """
    Computes SVI(t) using Sigmoid Late Fusion:
    SVI(t) = sigma(w_a * A(t) + w_t * T(t) + w_b * B(t) + delta_critical) * 100

    Deterministic Hard Override:
    If p_suicide > 0.80, p_panic > 0.80, or explicit markers (rape, weapon, murder),
    delta_critical = 0.50, forcing score straight to Critical Risk (81-100).

    Returns:
    (raw_svi, delta_critical, category, primary_trigger)
    """
    # Check deterministic hard override
    is_hard_override = (p_suicide > 0.80) or (p_panic > 0.80) or has_explicit_marker
    delta_critical = 0.50 if is_hard_override else 0.0

    # Weighted sum
    x = (
        WEIGHTS["acoustic"] * a_t
        + WEIGHTS["text"] * t_t
        + WEIGHTS["behavior"] * b_t
        + delta_critical
    )

    # Sigmoid scaling
    raw_svi = sigmoid_late_fusion(x) * 100.0

    # If hard override was triggered, guarantee landing in 81-100
    if is_hard_override and raw_svi < 81.0:
        raw_svi = 85.0

    category = get_category(raw_svi)
    trigger = "Deterministic Override Triggered" if is_hard_override else "Multimodal SVI Fusion"
    return float(raw_svi), float(delta_critical), category, trigger


def get_category(svi: float) -> str:
    if svi >= 81.0:
        return "CRITICAL_RISK"
    elif svi >= 51.0:
        return "HIGH_RISK"
    elif svi >= 26.0:
        return "MODERATE_RISK"
    else:
        return "LOW_RISK"


class EmaSmoother:
    """
    Exponential Moving Average smoother per spec:
    SVI_smooth(t) = lambda * SVI(t) + (1 - lambda) * SVI_smooth(t-1)
    with lambda = 0.3.
    """
    def __init__(self, initial: float = 10.0, lam: float = 0.3):
        self.value = initial
        self.lam = lam

    def update(self, target: float) -> float:
        self.value = self.lam * target + (1.0 - self.lam) * self.value
        return self.value

    def force(self, value: float) -> float:
        """Instant snap for hard trigger overrides."""
        self.value = value
        return self.value
