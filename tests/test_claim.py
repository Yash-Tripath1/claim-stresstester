from pathlib import Path

from stresstest.claim import ResearchClaim, load_claim, parse_claim_text

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "tutoring_claim.json"


def test_load_claim_roundtrip():
    claim = load_claim(FIXTURE)
    assert claim.study_type == "observational_cohort"
    assert claim.effect_estimate == 5.0
    assert claim.n_per_arm == 200
    assert claim.baseline_outcome == 60.0
    assert claim.effect_ci == (1.1, 8.9)
    d = claim.to_dict()
    assert d["effect_ci"] == [1.1, 8.9]
    claim2 = ResearchClaim.from_dict(d)
    assert claim2 == claim


def test_parse_claim_text_basic():
    text = (
        "The tutoring program improved scores by 5 points (p = 0.02, N = 200 per group). "
        "Observational cohort. Baseline 60, SD 12."
    )
    claim = parse_claim_text(text)
    assert claim.effect_estimate == 5.0
    assert claim.study_type == "observational_cohort"
    assert claim.n_per_arm == 200
    assert claim.baseline_outcome == 60.0
    assert claim.outcome_sd == 12.0
