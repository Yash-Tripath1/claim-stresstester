"""Research-claim representation and loading.

A ``ResearchClaim`` captures the minimum information the simulators need:
the reported effect, its significance, the sample size, and the outcome scale
(baseline mean and SD) that the synthetic data-generating process requires.

Inputs come from a JSON fixture (authoritative) or, best-effort, from free text.
Nothing here makes network calls.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class ResearchClaim:
    """A single research claim plus the scale info needed for simulation."""

    study_type: str
    effect_estimate: float
    p_value: float
    n_per_arm: int
    outcome_unit: str
    baseline_outcome: float
    outcome_sd: float
    raw_text: str
    effect_ci: Optional[Tuple[float, float]] = None
    source_label: str = "synthetic"

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.effect_ci is not None:
            d["effect_ci"] = list(self.effect_ci)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ResearchClaim":
        d = dict(d)
        if d.get("effect_ci") is not None:
            d["effect_ci"] = tuple(d["effect_ci"])
        known = set(cls.__dataclass_fields__)
        d = {k: v for k, v in d.items() if k in known}
        return cls(**d)


def load_claim(path: str | Path) -> ResearchClaim:
    """Load a claim from a JSON fixture file."""
    data = json.loads(Path(path).read_text())
    return ResearchClaim.from_dict(data)


_DEFAULTS = dict(
    baseline_outcome=60.0,
    outcome_sd=12.0,
    outcome_unit="points",
    study_type="observational_cohort",
)


def _first_float(text: str, patterns) -> Optional[float]:
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except (ValueError, IndexError):
                continue
    return None


def _first_int(text: str, patterns) -> Optional[int]:
    v = _first_float(text, patterns)
    return int(v) if v is not None else None


def _parse_ci(text: str) -> Optional[Tuple[float, float]]:
    m = re.search(
        r"(?:95\s*%?\s*)?(?:CI|confidence)[^0-9\-–]{0,10}"
        r"([+-]?\d+(?:\.\d+)?)\s*[–\-]\s*([+-]?\d+(?:\.\d+)?)",
        text,
        re.IGNORECASE,
    )
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        return (min(a, b), max(a, b))
    return None


def parse_claim_text(text: str) -> ResearchClaim:
    """Best-effort regex parser. Used only when no JSON fixture is supplied."""
    low = text.lower()
    if "randomi" in low or "rct" in low:
        study_type = "randomized_trial"
    elif "cohort" in low or "observation" in low:
        study_type = "observational_cohort"
    else:
        study_type = _DEFAULTS["study_type"]

    effect = _first_float(
        text,
        [
            r"([+-]?\d+(?:\.\d+)?)\s*(?:point|pts?|score|improv|increase|gain|effect)",
            r"effect\s*(?:of|was|=|:)?\s*([+-]?\d+(?:\.\d+)?)",
        ],
    )
    if effect is None:
        effect = _DEFAULTS["baseline_outcome"] * 0.0
    p = _first_float(text, [r"p\s*(?:<|=|value\s*=)?\s*([0-9]*\.?[0-9]+)"]) or 1.0
    n = _first_int(text, [r"\bn\s*(?:=|of|per)?\s*(\d{2,5})"]) or 200
    ci = _parse_ci(text)
    baseline = (
        _first_float(text, [r"baseline[^0-9]{0,15}([0-9]+(?:\.[0-9]+)?)"])
        or _DEFAULTS["baseline_outcome"]
    )
    sd = (
        _first_float(text, [r"(?:sd|standard deviation)[^0-9]{0,10}([0-9]+(?:\.[0-9]+)?)"])
        or _DEFAULTS["outcome_sd"]
    )
    unit = "points"
    m = re.search(r"([a-z]+)\s*\)", text)  # crude unit hint
    if m and m.group(1).lower() not in {"p", "ci", "n"}:
        unit = m.group(1).lower()

    return ResearchClaim(
        study_type=study_type,
        effect_estimate=float(effect),
        p_value=float(min(1.0, max(0.0, p))),
        n_per_arm=int(n),
        outcome_unit=unit,
        baseline_outcome=float(baseline),
        outcome_sd=float(sd),
        raw_text=text,
        effect_ci=ci,
        source_label="parsed_from_text",
    )
