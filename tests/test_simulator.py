import numpy as np

from stresstest.claim import load_claim
from stresstest.templates import get_template
from stresstest.simulator import Simulator, welch_p, POSSIBLE, REJECTED

FIXTURE = "fixtures/tutoring_claim.json"


def test_welch_p_basics():
    a = np.array([10.0] * 50)
    b = np.array([20.0] * 50)
    assert welch_p(a, b) < 0.001
    assert welch_p(a, a.copy()) == 1.0


def test_strong_confounder_is_possible():
    claim = load_claim(FIXTURE)
    t = get_template("unmeasured_confounding")
    sim = Simulator(n_replicates=600, seed=12345)
    res = sim.evaluate(claim, t, {"prevalence_u": 0.30, "rr_ud": 3.0, "rr_eu": 1.22})
    assert res.verdict == POSSIBLE
    assert res.fraction_matching_claim > 0.10
    assert abs(res.apparent_effect_mean - 5.0) < 2.0


def test_weak_confounder_is_rejected():
    claim = load_claim(FIXTURE)
    t = get_template("unmeasured_confounding")
    sim = Simulator(n_replicates=600, seed=12345)
    res = sim.evaluate(claim, t, {"prevalence_u": 0.05, "rr_ud": 1.1, "rr_eu": 1.01})
    assert res.verdict == REJECTED
    assert res.fraction_matching_claim <= 0.02


def test_attrition_with_selection_is_possible():
    claim = load_claim(FIXTURE)
    t = get_template("differential_attrition")
    sim = Simulator(n_replicates=600, seed=12345)
    res = sim.evaluate(
        claim,
        t,
        {
            "attrition_rate_trt": 0.30,
            "attrition_rate_ctrl": 0.10,
            "attrite_outcome_trt_offset": -12.0,
            "attrite_outcome_ctrl_offset": 0.0,
        },
    )
    assert res.verdict == POSSIBLE
    assert abs(res.apparent_effect_mean - 5.0) < 2.0


def test_attrition_without_selection_is_rejected():
    claim = load_claim(FIXTURE)
    t = get_template("differential_attrition")
    sim = Simulator(n_replicates=600, seed=12345)
    res = sim.evaluate(
        claim,
        t,
        {
            "attrition_rate_trt": 0.20,
            "attrition_rate_ctrl": 0.10,
            "attrite_outcome_trt_offset": 0.0,
            "attrite_outcome_ctrl_offset": 0.0,
        },
    )
    assert res.verdict == REJECTED


def test_result_to_dict():
    claim = load_claim(FIXTURE)
    t = get_template("unmeasured_confounding")
    sim = Simulator(n_replicates=200, seed=1)
    res = sim.evaluate(claim, t, {"prevalence_u": 0.30, "rr_ud": 3.0, "rr_eu": 1.22})
    d = res.to_dict()
    assert d["template_id"] == "unmeasured_confounding"
    assert isinstance(d["apparent_effect_ci"], list)
