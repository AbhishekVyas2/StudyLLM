"""
Tests for adaptive performance profiles.
"""

from study_llm.hardware.detector import PerformanceTier
from study_llm.performance.profile import (
    PerformanceProfile, get_performance_profile, apply_profile_to_config
)


def make_tier(n: int) -> PerformanceTier:
    sizes = {1: 1.5, 2: 3, 3: 8, 4: 14, 5: 30}
    return PerformanceTier(
        tier=n,
        description=f"tier{n}",
        recommended_model_size_billions=sizes[n],
        notes="",
    )


def test_profile_per_tier():
    p1 = get_performance_profile(make_tier(1))
    p5 = get_performance_profile(make_tier(5))
    assert p1.profile_name == "lite"
    assert p5.profile_name == "quality"
    assert p1.context_budget_tokens < p5.context_budget_tokens
    assert not p1.use_reranker and p5.use_reranker
    assert isinstance(p1, PerformanceProfile)


def test_apply_profile_respects_user_overrides():
    from study_llm.core.config import StudyLLMConfig

    # User explicitly set ctx_size=8192 on low-end hw -> keep it
    cfg = StudyLLMConfig()
    cfg.ctx_size = 8192
    profile = apply_profile_to_config(cfg)
    assert cfg.ctx_size == 8192

    # Non-overridden values get tier defaults
    assert cfg.retrieval_top_k == profile.retrieval_top_k


def test_apply_profile_fills_defaults():
    from study_llm.core.config import StudyLLMConfig
    cfg = StudyLLMConfig()
    profile = apply_profile_to_config(cfg)
    assert cfg.ctx_size == profile.ctx_size
    assert cfg.context_budget_tokens == profile.context_budget_tokens
