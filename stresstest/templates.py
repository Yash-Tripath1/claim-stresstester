"""Bounded rival-explanation templates.

The tool deliberately offers only a small, declared set of alternative
explanations. Each template fixes the *shape* of the explanation and exposes a
small, bounded set of parameters the proposer may fill in. This prevents the
model from free-generating arbitrary "alternatives" (a known failure mode of
LLM critics) and keeps every check auditable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class ParamSpec:
    name: str
    type: str  # "float" | "int"
    low: float
    high: float
    description: str
    default: float


@dataclass(frozen=True)
class RivalTemplate:
    id: str
    name: str
    kind: str  # "confounding" | "attrition"
    description: str
    applies_to: Tuple[str, ...]
    params: Tuple[ParamSpec, ...]

    def param_names(self) -> List[str]:
        return [p.name for p in self.params]

    def default_params(self) -> Dict[str, float]:
        return {p.name: p.default for p in self.params}

    def validate_and_clip(self, params: Dict[str, float]) -> Dict[str, float]:
        """Coerce types, clip to bounds, raise ValueError on non-numeric input."""
        out: Dict[str, float] = {}
        for spec in self.params:
            if spec.name not in params:
                out[spec.name] = spec.default
                continue
            raw = params[spec.name]
            if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
                raise ValueError(f"param '{spec.name}' must be numeric, got {raw!r}")
            try:
                val = float(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"param '{spec.name}' must be numeric, got {raw!r}") from exc
            if not math_isfinite(val):
                raise ValueError(f"param '{spec.name}' must be finite, got {val!r}")
            val = max(spec.low, min(spec.high, val))
            if spec.type == "int":
                val = int(round(val))
            out[spec.name] = val
        return out


def math_isfinite(x: float) -> bool:
    import math

    return math.isfinite(x)


_UNMEASURED_CONFOUNDING = RivalTemplate(
    id="unmeasured_confounding",
    name="Unmeasured confounding",
    kind="confounding",
    description=(
        "An unmeasured binary factor U is associated with both treatment "
        "receipt and outcome. If U is imbalanced between the compared groups, "
        "it can manufacture an apparent treatment effect even when the true "
        "effect is zero."
    ),
    applies_to=("observational_cohort", "observational_case_control", "observational"),
    params=(
        ParamSpec(
            "prevalence_u", "float", 0.01, 0.6,
            "Fraction of the sample carrying the unmeasured confounder U.",
            0.20,
        ),
        ParamSpec(
            "rr_ud", "float", 0.5, 6.0,
            "Risk ratio U->treatment: how much more likely U=1 subjects are to "
            "receive treatment (vs U=0).",
            2.0,
        ),
        ParamSpec(
            "rr_eu", "float", 0.9, 3.0,
            "Risk ratio U->outcome: multiplier on the conditional mean outcome "
            "for U=1 vs U=0 (e.g. 1.20 means U raises the mean by 20%).",
            1.10,
        ),
    ),
)

_DIFFERENTIAL_ATTRITION = RivalTemplate(
    id="differential_attrition",
    name="Differential attrition",
    kind="attrition",
    description=(
        "Participants lost to follow-up differ between arms and are not "
        "representative of their arm. Removing them can change the observed "
        "mean difference even when the true effect is zero."
    ),
    applies_to=(
        "observational_cohort",
        "randomized_trial",
        "observational_case_control",
        "observational",
    ),
    params=(
        ParamSpec(
            "attrition_rate_trt", "float", 0.0, 0.8,
            "Fraction of the treatment arm lost to follow-up.",
            0.20,
        ),
        ParamSpec(
            "attrition_rate_ctrl", "float", 0.0, 0.8,
            "Fraction of the control arm lost to follow-up.",
            0.10,
        ),
        ParamSpec(
            "attrite_outcome_trt_offset", "float", -30.0, 30.0,
            "Mean outcome among treatment-arm dropouts, relative to baseline "
            "(negative = low performers drop out).",
            -8.0,
        ),
        ParamSpec(
            "attrite_outcome_ctrl_offset", "float", -30.0, 30.0,
            "Mean outcome among control-arm dropouts, relative to baseline.",
            0.0,
        ),
    ),
)

TEMPLATES: Dict[str, RivalTemplate] = {
    _UNMEASURED_CONFOUNDING.id: _UNMEASURED_CONFOUNDING,
    _DIFFERENTIAL_ATTRITION.id: _DIFFERENTIAL_ATTRITION,
}


def get_template(template_id: str) -> RivalTemplate:
    if template_id not in TEMPLATES:
        raise KeyError(
            f"Unknown template '{template_id}'. Available: {sorted(TEMPLATES)}"
        )
    return TEMPLATES[template_id]


def list_templates() -> List[RivalTemplate]:
    return list(TEMPLATES.values())


def applicable_templates(study_type: str) -> List[RivalTemplate]:
    return [t for t in TEMPLATES.values() if study_type in t.applies_to]
