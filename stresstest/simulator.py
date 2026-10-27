"""Deterministic, seeded Monte-Carlo stress-test simulator.

Given a claim and a rival template's parameters, the simulator generates many
synthetic datasets under the *alternative explanation's* assumptions (true
effect = 0) and checks how often the naive observed effect reproduces the
claimed result (effect magnitude within tolerance AND p < 0.05).

Verdicts:
  * POSSIBLE_UNDER_ASSUMPTIONS  -- the alternative can reproduce the result
    (matching fraction >= 0.10).
  * REJECTED_UNDER_ASSUMPTIONS  -- the alternative essentially cannot
    (matching fraction <= 0.02).
  * UNCERTAIN                   -- in between.

The verdict is a statement about *modelled plausibility*, never about truth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .claim import ResearchClaim
from .templates import RivalTemplate

POSSIBLE = "POSSIBLE_UNDER_ASSUMPTIONS"
REJECTED = "REJECTED_UNDER_ASSUMPTIONS"
UNCERTAIN = "UNCERTAIN"

# Internal modelling constant: base treatment probability among U=0 subjects.
# Documented as an assumption in every result; not a proposable parameter.
_BASE_TREATMENT_PROB_U0 = 0.30


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def welch_p(a, b) -> float:
    """Two-sided Welch t-test p-value via a normal approximation (large n)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = a.size, b.size
    if na < 2 or nb < 2:
        return 1.0
    ma, mb = float(a.mean()), float(b.mean())
    va, vb = float(a.var(ddof=1)), float(b.var(ddof=1))
    se = math.sqrt(va / na + vb / nb)
    if se == 0.0:
        return 0.0 if ma != mb else 1.0
    t = abs(ma - mb) / se
    return min(1.0, max(0.0, 2.0 * (1.0 - _norm_cdf(t))))


@dataclass
class SimulationResult:
    template_id: str
    verdict: str
    parameters: Dict[str, float]
    apparent_effect_mean: float
    apparent_effect_ci: Tuple[float, float]
    fraction_matching_claim: float
    n_replicates: int
    seed: int
    assumptions: Dict[str, object]
    notes: str

    def to_dict(self) -> dict:
        return {
            "template_id": self.template_id,
            "verdict": self.verdict,
            "parameters": dict(self.parameters),
            "apparent_effect_mean": self.apparent_effect_mean,
            "apparent_effect_ci": list(self.apparent_effect_ci),
            "fraction_matching_claim": self.fraction_matching_claim,
            "n_replicates": self.n_replicates,
            "seed": self.seed,
            "assumptions": dict(self.assumptions),
            "notes": self.notes,
        }


class Simulator:
    def __init__(self, n_replicates: int = 1000, seed: int = 12345):
        self.n_replicates = n_replicates
        self.seed = seed

    # -- public API --------------------------------------------------------
    def evaluate(
        self, claim: ResearchClaim, template: RivalTemplate, params: Dict[str, float]
    ) -> SimulationResult:
        params = template.validate_and_clip(params)
        if template.kind == "confounding":
            return self._eval_confounding(claim, params)
        if template.kind == "attrition":
            return self._eval_attrition(claim, params)
        raise ValueError(f"Unknown template kind '{template.kind}'")

    # -- helpers -----------------------------------------------------------
    def _verdict(self, frac: float) -> str:
        if frac >= 0.10:
            return POSSIBLE
        if frac <= 0.02:
            return REJECTED
        return UNCERTAIN

    def _finish(
        self,
        template: RivalTemplate,
        params: Dict[str, float],
        effects: np.ndarray,
        ps: np.ndarray,
        claim: ResearchClaim,
        assumptions: Dict[str, object],
        notes_fn,
    ) -> SimulationResult:
        tol = max(1.0, 0.3 * abs(claim.effect_estimate))
        lo, hi = claim.effect_estimate - tol, claim.effect_estimate + tol
        match = (effects >= lo) & (effects <= hi) & (ps < 0.05)
        frac = float(match.mean()) if effects.size else 0.0
        ci = (float(np.percentile(effects, 2.5)), float(np.percentile(effects, 97.5)))
        verdict = self._verdict(frac)
        return SimulationResult(
            template_id=template.id,
            verdict=verdict,
            parameters=dict(params),
            apparent_effect_mean=float(effects.mean()) if effects.size else 0.0,
            apparent_effect_ci=ci,
            fraction_matching_claim=frac,
            n_replicates=int(effects.size),
            seed=self.seed,
            assumptions=assumptions,
            notes=notes_fn(verdict, frac),
        )

    # -- confounding ------------------------------------------------------
    def _eval_confounding(
        self, claim: ResearchClaim, p: Dict[str, float]
    ) -> SimulationResult:
        template = _template_for("confounding")
        n = claim.n_per_arm
        prev = p["prevalence_u"]
        rr_ud = p["rr_ud"]
        rr_eu = p["rr_eu"]
        baseline = claim.baseline_outcome
        sd = claim.outcome_sd

        p_t0 = _BASE_TREATMENT_PROB_U0
        p_t1 = min(rr_ud * p_t0, 0.98)
        p_t1 = max(p_t1, 1e-6)
        p_t0 = max(p_t0, 1e-6)
        p_t0 = min(p_t0, 1 - 1e-6)
        p_t1 = min(p_t1, 1 - 1e-6)
        p_T1 = prev * p_t1 + (1 - prev) * p_t0
        p_U_given_T1 = (prev * p_t1 / p_T1) if p_T1 > 0 else prev
        p_U_given_T0 = (prev * p_t0 / (1 - p_T1)) if (1 - p_T1) > 0 else 0.0
        p_U_given_T1 = min(max(p_U_given_T1, 0.0), 1.0)
        p_U_given_T0 = min(max(p_U_given_T0, 0.0), 1.0)
        delta_u = (rr_eu - 1.0) * baseline

        effects = np.empty(self.n_replicates)
        ps = np.empty(self.n_replicates)
        for i in range(self.n_replicates):
            rng = np.random.default_rng(self.seed * 100003 + i)
            u1 = rng.binomial(1, p_U_given_T1, n)
            u0 = rng.binomial(1, p_U_given_T0, n)
            y1 = baseline + delta_u * u1 + rng.normal(0.0, sd, n)
            y0 = baseline + delta_u * u0 + rng.normal(0.0, sd, n)
            effects[i] = y1.mean() - y0.mean()
            ps[i] = welch_p(y1, y0)

        assumptions = {
            "true_effect": 0.0,
            "confounder": "binary U, unmeasured",
            "prevalence_u": prev,
            "rr_ud": rr_ud,
            "rr_eu": rr_eu,
            "outcome_model": "additive shift delta_u = (rr_eu - 1) * baseline",
            "treatment_base_rate_u0": p_t0,
            "noise": f"N(0, {sd})",
            "match_band": "effect within max(1, 0.3*|claim|) of claim AND p<0.05",
        }

        def notes(verdict: str, frac: float) -> str:
            if verdict == POSSIBLE:
                return (
                    f"Under a zero true effect and the stated confounder "
                    f"assumptions, {frac:.0%} of simulated datasets reproduce a "
                    f"result consistent with the claim. The alternative is "
                    f"POSSIBLE under these assumptions (not proven true)."
                )
            if verdict == REJECTED:
                return (
                    f"Under the stated confounder assumptions the alternative "
                    f"reproduces the claimed result in only {frac:.1%} of "
                    f"simulations -- it is unlikely to explain the result."
                )
            return (
                f"Ambiguous: the alternative reproduces the claimed result in "
                f"{frac:.1%} of simulations, between the possible/rejected bands."
            )

        return self._finish(template, p, effects, ps, claim, assumptions, notes)

    # -- attrition --------------------------------------------------------
    def _eval_attrition(
        self, claim: ResearchClaim, p: Dict[str, float]
    ) -> SimulationResult:
        template = _template_for("attrition")
        n = claim.n_per_arm
        a_t = p["attrition_rate_trt"]
        a_c = p["attrition_rate_ctrl"]
        off_t = p["attrite_outcome_trt_offset"]
        off_c = p["attrite_outcome_ctrl_offset"]
        baseline = claim.baseline_outcome
        sd = claim.outcome_sd

        def inflation(off: float, a: float) -> float:
            if a >= 1.0:
                return 0.0
            return -off * a / (1.0 - a)

        infl_t = inflation(off_t, a_t)
        infl_c = inflation(off_c, a_c)
        n_comp_t = max(1, int(round(n * (1.0 - a_t))))
        n_comp_c = max(1, int(round(n * (1.0 - a_c))))

        effects = np.empty(self.n_replicates)
        ps = np.empty(self.n_replicates)
        for i in range(self.n_replicates):
            rng = np.random.default_rng(self.seed * 100003 + i)
            y_comp_t = rng.normal(baseline + infl_t, sd, n_comp_t)
            y_comp_c = rng.normal(baseline + infl_c, sd, n_comp_c)
            effects[i] = y_comp_t.mean() - y_comp_c.mean()
            ps[i] = welch_p(y_comp_t, y_comp_c)

        assumptions = {
            "true_effect": 0.0,
            "attrition_rate_trt": a_t,
            "attrition_rate_ctrl": a_c,
            "attrite_outcome_trt_offset": off_t,
            "attrite_outcome_ctrl_offset": off_c,
            "completers_inflation_trt": infl_t,
            "completers_inflation_ctrl": infl_c,
            "note": (
                "Full-arm mean is preserved at baseline; low-outcome dropouts "
                "inflate the observed completers' mean."
            ),
            "match_band": "effect within max(1, 0.3*|claim|) of claim AND p<0.05",
        }

        def notes(verdict: str, frac: float) -> str:
            if verdict == POSSIBLE:
                return (
                    f"With the stated dropout pattern and a zero true effect, "
                    f"{frac:.0%} of simulated datasets reproduce a result "
                    f"consistent with the claim. Possible under assumptions."
                )
            if verdict == REJECTED:
                return (
                    f"The stated dropout pattern reproduces the claimed result in "
                    f"only {frac:.1%} of simulations -- unlikely to explain it."
                )
            return (
                f"Ambiguous: {frac:.1%} of simulations reproduce the claimed "
                f"result, between the possible/rejected bands."
            )

        return self._finish(template, p, effects, ps, claim, assumptions, notes)


# Avoid a circular import at module load by resolving the template lazily.
def _template_for(kind: str) -> RivalTemplate:
    from .templates import get_template

    if kind == "confounding":
        return get_template("unmeasured_confounding")
    if kind == "attrition":
        return get_template("differential_attrition")
    raise ValueError(f"Unknown kind '{kind}'")
