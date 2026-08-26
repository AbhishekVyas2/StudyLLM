"""
Adaptive performance profiles for StudyLLM.
Derives runtime settings from detected hardware (tiers 1-5).
"""

import logging
from dataclasses import dataclass

from study_llm.hardware.detector import get_performance_tier, PerformanceTier

logger = logging.getLogger(__name__)


@dataclass
class PerformanceProfile:
    """Hardware-derived runtime settings."""
    profile_name: str            # lite | balanced | quality
    ctx_size: int
    gpu_layers: int
    retrieval_top_k: int
    reranker_top_k: int
    context_budget_tokens: int
    max_tokens: int
    use_reranker: bool


# Per-tier settings. Low-end machines get smaller contexts, no reranker,
# and tighter budgets so answers stay fast.
# Tier 3 now requires 6+ GB VRAM to be "Mainstream" (see detector.py), so its
# profile can afford a bigger context than pure-CPU tiers without slowdown.
_TIER_PROFILES = {
    1: dict(profile_name="lite",     ctx_size=2048,  gpu_layers=0,
            retrieval_top_k=5,  reranker_top_k=0, context_budget_tokens=800,
            max_tokens=256,  use_reranker=False),
    2: dict(profile_name="lite",     ctx_size=2048,  gpu_layers=0,
            retrieval_top_k=6,  reranker_top_k=0, context_budget_tokens=1000,
            max_tokens=384,  use_reranker=False),
    3: dict(profile_name="balanced", ctx_size=4096,  gpu_layers=0,
            retrieval_top_k=8,  reranker_top_k=3, context_budget_tokens=1500,
            max_tokens=512,  use_reranker=False),
    4: dict(profile_name="quality",  ctx_size=8192,  gpu_layers=16,
            retrieval_top_k=10, reranker_top_k=3, context_budget_tokens=2500,
            max_tokens=768,  use_reranker=True),
    5: dict(profile_name="quality",  ctx_size=16384, gpu_layers=32,
            retrieval_top_k=10, reranker_top_k=5, context_budget_tokens=3000,
            max_tokens=1024, use_reranker=True),
}


def get_performance_profile(tier=None) -> PerformanceProfile:
    """Map the hardware tier to a performance profile."""
    if tier is None:
        tier = get_performance_tier()
    settings = _TIER_PROFILES[tier.tier]
    return PerformanceProfile(**settings)


def apply_profile_to_config(config) -> PerformanceProfile:
    """Apply tier-appropriate defaults to a config (user overrides win)."""
    profile = get_performance_profile()

    # Only fill in values the user hasn't overridden explicitly
    if config.ctx_size == 2048:
        config.ctx_size = profile.ctx_size
    if config.gpu_layers == 0:
        config.gpu_layers = profile.gpu_layers
    if config.retrieval_top_k == 10:
        config.retrieval_top_k = profile.retrieval_top_k
    if config.reranker_top_k == 3:
        config.reranker_top_k = profile.reranker_top_k
    if config.context_budget_tokens == 1500:
        config.context_budget_tokens = profile.context_budget_tokens

    return profile
