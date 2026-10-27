from stresstest.claim import load_claim
from stresstest.templates import get_template
from stresstest.proposer import DeterministicFallbackProposer
from stresstest.simulator import Simulator
from stresstest.discrimination import propose_followup, power_check
from stresstest.evidence import render_markdown, render_html, DISCLAIMER

FIXTURE = "fixtures/tutoring_claim.json"


def _card():
    claim = load_claim(FIXTURE)
    proposer = DeterministicFallbackProposer()
    t = get_template("unmeasured_confounding")
    prop = proposer.propose(claim, t)
    sim = Simulator(n_replicates=300, seed=1).evaluate(claim, t, prop.parameters)
    fu = propose_followup(claim, t, prop.parameters)
    disc = power_check(claim, t, prop.parameters, fu, n_replicates=300, seed=987654)
    return claim, prop, sim, fu, disc


def test_markdown_contains_key_sections():
    claim, prop, sim, fu, disc = _card()
    md = render_markdown(claim, prop, sim, fu, disc)
    assert "Evidence Card" in md
    assert "Simulation result" in md
    assert "Assumptions" in md
    assert "What this does NOT prove" in md
    assert DISCLAIMER in md
    assert sim.verdict in md


def test_html_contains_disclaimer():
    claim, prop, sim, fu, disc = _card()
    html = render_html(claim, prop, sim, fu, disc)
    assert "<html" in html
    assert "What this does NOT prove" in html
    assert DISCLAIMER in html


def test_markdown_handles_rejected_without_followup():
    claim = load_claim(FIXTURE)
    proposer = DeterministicFallbackProposer()
    t = get_template("unmeasured_confounding")
    prop = proposer.propose(claim, t)
    sim = Simulator(n_replicates=300, seed=1).evaluate(
        claim, t, {"prevalence_u": 0.05, "rr_ud": 1.1, "rr_eu": 1.01}
    )
    md = render_markdown(claim, prop, sim, None, None)
    assert "Not run" in md
    assert DISCLAIMER in md
