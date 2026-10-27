"""Follow-up proposal and deterministic discrimination-power check.

Once an alternative explanation is ``POSSIBLE_UNDER_ASSUMPTIONS``, the tool
proposes ONE follow-up experiment that could distinguish the alternative from
the original claim, and deterministically checks whether that follow-up would
actually discriminate.

The check is a seeded Monte-Carlo comparison of two worlds:

  * World A -- the alternative explanation is true (the apparent result is an
    artifact; the "true" effect behind it is zero).
  * World B -- the original claim is true (the effect is real).

We compute the follow-up's test statistic in both worlds and measure how
often a simple decision rule separates them. Verdict:

  * DISCRIMINATES           -- power >= 0.80
  * WEAK                    -- power >= 0.60
  * DOES_NOT_DISCRIMINATE   -- power <  0.60
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .claim import ResearchClaim
from .templates import RivalTemplate

DISCRIMINATES = "DISCRIMINATES"
WEAK = "WEAK"
DOES_NOT = "DOES_NOT_DISCRIMINATE"


@dataclass
class FollowupProposal:
    design: str
    test: str
    sample_size: int
    rationale: str

    def to_dict(self) -> dict:
        return {
            "design": self.design,
            "test": self.test,
            "sample_size": self.sample_size,
            "rationale": self.rationale,
        }


@dataclass
class DiscriminationResult:
    verdict: str
    power: float
    power_world_A: float
    power_world_B: float
    expected_effect_world_A: float
    expected_effect_world_B: float
    se: float
    n_follow: int
    method: str
    notes: str

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "power": self.power,
            "power_world_A": self.power_world_A,
            "power_world_B": self.power_world_B,
            "expected_effect_world_A": self.expected_effect_world_A,
            "expected_effect_world_B": self.expected_effect_world_B,
            "se": self.se,
            "n_follow": self.n_follow,
            "method": self.method,
            "notes": self.notes,
        }


def propose_followup(
    claim: ResearchClaim, template: RivalTemplate, params: Dict[str, float]
) -> FollowupProposal:
    if template.id == "unmeasured_confounding":
        return FollowupProposal(
            design=(
                "New cohort in which the confounder U (or a validated proxy) is "
                "measured; re-estimate the treatment effect adjusting for U."
            ),
            test="Adjusted (U-controlled) treatment effect",
            sample_size=300,
            rationale=(
                "Adjusting for U breaks the confounding path. If the apparent "
                "effect vanishes under adjustment, the confounding explanation "
                "is supported; if it persists, the original claim is supported."
            ),
        )
    if template.id == "differential_attrition":
        return FollowupProposal(
            design=(
                "Recover outcomes for all attriters (active follow-up) and "
                "re-estimate an intent-to-treat effect on the full cohort."
            ),
            test="Full-cohort (recovered) treatment effect",
            sample_size=300,
            rationale=(
                "If the effect disappears once attriters are recovered, the "
                "attrition explanation is supported; if it persists, the "
                "original claim is supported."
            ),
        )
    raise ValueError(f"No follow-up defined for template '{template.id}'")


def _ols_treatment_coef(t, u, y) -> float:
    """Coefficient on T from Y ~ 1 + T + U (OLS)."""
    x = np.column_stack([np.ones_like(t, dtype=float), t.astype(float), u.astype(float)])
    coef, *_ = np.linalg.lstsq(x, y.astype(float), rcond=None)
    return float(coef[1])


def _verdict(power: float) -> str:
    if power >= 0.80:
        return DISCRIMINATES
    if power >= 0.60:
        return WEAK
    return DOES_NOT


def _summarize(
    template: RivalTemplate,
    n_reps: int,
    seed: int,
    coefs_a: np.ndarray,
    coefs_b: np.ndarray,
    world_a_label: str,
    world_b_label: str,
) -> DiscriminationResult:
    mid = 0.5 * (float(coefs_a.mean()) + float(coefs_b.mean()))
    # Decision rule: above mid -> "claim true"; below mid -> "alternative true".
    power_a = float((coefs_a <= mid).mean())
    power_b = float((coefs_b > mid).mean())
    power = 0.5 * (power_a + power_b)
    se_a = float(coefs_a.std(ddof=1)) if coefs_a.size > 1 else 0.0
    se_b = float(coefs_b.std(ddof=1)) if coefs_b.size > 1 else 0.0
    verdict = _verdict(power)
    notes = (
        f"Decision rule threshold = {mid:.2f}. "
        f"World A ('{world_a_label}') correctly classified {power_a:.1%} of the "
        f"time; World B ('{world_b_label}') {power_b:.1%}. "
        f"Combined discrimination power = {power:.1%}."
    )
    return DiscriminationResult(
        verdict=verdict,
        power=power,
        power_world_A=power_a,
        power_world_B=power_b,
        expected_effect_world_A=float(coefs_a.mean()),
        expected_effect_world_B=float(coefs_b.mean()),
        se=max(se_a, se_b),
        n_follow=0,
        method="seeded Monte-Carlo two-world comparison",
        notes=notes,
    )


def power_check(
    claim: ResearchClaim,
    template: RivalTemplate,
    params: Dict[str, float],
    followup: FollowupProposal,
    n_replicates: int = 1000,
    seed: int = 987654,
) -> DiscriminationResult:
    params = template.validate_and_clip(params)
    n = max(2, int(followup.sample_size))

    if template.id == "unmeasured_confounding":
        return _power_confounding(claim, params, n, n_replicates, seed)
    if template.id == "differential_attrition":
        return _power_attrition(claim, params, n, n_replicates, seed)
    raise ValueError(f"No discrimination check for template '{template.id}'")


def _power_confounding(claim, p, n, n_reps, seed):
    prev = p["prevalence_u"]
    rr_ud = p["rr_ud"]
    rr_eu = p["rr_eu"]
    baseline = claim.baseline_outcome
    sd = claim.outcome_sd
    true_effect = claim.effect_estimate

    p_t0 = 0.30
    p_t1 = min(rr_ud * p_t0, 0.98)
    p_t1 = max(min(p_t1, 1 - 1e-6), 1e-6)
    p_T1 = prev * p_t1 + (1 - prev) * p_t0
    delta_u = (rr_eu - 1.0) * baseline

    coefs_a = np.empty(n_reps)
    coefs_b = np.empty(n_reps)
    for i in range(n_reps):
        rng = np.random.default_rng(seed * 100003 + i)
        u = rng.binomial(1, prev, n)
        # World A: confounding true, true treatment effect = 0.
        t_a = rng.binomial(1, np.where(u == 1, p_t1, p_t0))
        y_a = baseline + delta_u * u + rng.normal(0.0, sd, n)
        # World B: claim true, treatment independent of U, true effect present.
        t_b = rng.binomial(1, 0.5, n)
        y_b = baseline + true_effect * t_b + rng.normal(0.0, sd, n)
        coefs_a[i] = _ols_treatment_coef(t_a, u, y_a)
        coefs_b[i] = _ols_treatment_coef(t_b, u, y_b)

    res = _summarize(
        _get("unmeasured_confounding"), n_reps, seed, coefs_a, coefs_b,
        "confounding explains the result", "claim is true",
    )
    res.n_follow = n
    res.notes += (
        " World A: adjusted effect centered near 0 (confounding removed). "
        "World B: adjusted effect centered near the reported effect."
    )
    return res


def _power_attrition(claim, p, n, n_reps, seed):
    baseline = claim.baseline_outcome
    sd = claim.outcome_sd
    true_effect = claim.effect_estimate

    effs_a = np.empty(n_reps)
    effs_b = np.empty(n_reps)
    for i in range(n_reps):
        rng = np.random.default_rng(seed * 100003 + i)
        # World A: full cohort true effect = 0 (attrition manufactured the gap).
        y_a_t = rng.normal(baseline, sd, n)
        y_a_c = rng.normal(baseline, sd, n)
        effs_a[i] = y_a_t.mean() - y_a_c.mean()
        # World B: true effect present, no attrition confounding.
        y_b_t = rng.normal(baseline + true_effect, sd, n)
        y_b_c = rng.normal(baseline, sd, n)
        effs_b[i] = y_b_t.mean() - y_b_c.mean()

    res = _summarize(
        _get("differential_attrition"), n_reps, seed, effs_a, effs_b,
        "attrition explains the result", "claim is true",
    )
    res.n_follow = n
    res.notes += (
        " World A: recovered full-cohort effect centered near 0. "
        "World B: recovered full-cohort effect centered near the reported effect."
    )
    return res


def _get(template_id: str) -> RivalTemplate:
    from .templates import get_template

    return get_template(template_id)
