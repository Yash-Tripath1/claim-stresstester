from stresstest.claim import load_claim
from stresstest.templates import get_template
from stresstest.proposer import DeterministicFallbackProposer
from stresstest.simulator import Simulator
from stresstest.discrimination import (
    propose_followup,
    power_check,
    DISCRIMINATES,
    WEAK,
    DOES_NOT,
)

FIXTURE = "fixtures/tutoring_claim.json"
VERDICTS = {DISCRIMINATES, WEAK, DOES_NOT}


def _claim_and_template():
    claim = load_claim(FIXTURE)
    proposer = DeterministicFallbackProposer()
    return claim, proposer


def test_confounding_followup_discriminates():
    claim, proposer = _claim_and_template()
    t = get_template("unmeasured_confounding")
    params = proposer.propose(claim, t).parameters
    fu = propose_followup(claim, t, params)
    disc = power_check(claim, t, params, fu, n_replicates=400, seed=987654)
    assert disc.verdict in VERDICTS
    assert 0.0 <= disc.power <= 1.0
    assert disc.n_follow == fu.sample_size
    # For this synthetic example the follow-up should be decisive.
    assert disc.verdict == DISCRIMINATES


def test_attrition_followup_discriminates():
    claim, proposer = _claim_and_template()
    t = get_template("differential_attrition")
    params = proposer.propose(claim, t).parameters
    fu = propose_followup(claim, t, params)
    disc = power_check(claim, t, params, fu, n_replicates=400, seed=987654)
    assert disc.verdict in VERDICTS
    assert 0.0 <= disc.power <= 1.0
    assert disc.verdict == DISCRIMINATES


def test_followup_proposal_fields():
    claim, proposer = _claim_and_template()
    t = get_template("unmeasured_confounding")
    fu = propose_followup(claim, t, {})
    assert fu.sample_size >= 2
    assert fu.design
    assert fu.test
    assert fu.rationale
