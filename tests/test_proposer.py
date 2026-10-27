import pytest

from stresstest.claim import load_claim
from stresstest.templates import get_template
from stresstest.proposer import DeterministicFallbackProposer, GuardedMetaProposer

FIXTURE = "fixtures/tutoring_claim.json"


def test_fallback_proposes_valid_params():
    claim = load_claim(FIXTURE)
    proposer = DeterministicFallbackProposer()
    for tid in ("unmeasured_confounding", "differential_attrition"):
        t = get_template(tid)
        prop = proposer.propose(claim, t)
        assert prop.template_id == tid
        assert prop.source == "offline_deterministic_fallback"
        clipped = t.validate_and_clip(prop.parameters)
        assert clipped == prop.parameters


def test_guarded_meta_proposer_refuses_when_disabled():
    claim = load_claim(FIXTURE)
    t = get_template("unmeasured_confounding")
    proposer = GuardedMetaProposer(provider="x", model="y", enabled=False)
    assert proposer.is_live is False
    with pytest.raises(RuntimeError):
        proposer.propose(claim, t)


def test_guarded_meta_proposer_enabled_but_no_live_path():
    claim = load_claim(FIXTURE)
    t = get_template("unmeasured_confounding")
    proposer = GuardedMetaProposer(
        provider="x", model="y", enabled=True, allow_live=True
    )
    assert proposer.is_live is True
    with pytest.raises(NotImplementedError):
        proposer.propose(claim, t)
