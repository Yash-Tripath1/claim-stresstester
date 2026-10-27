import pytest

from stresstest.templates import get_template, list_templates, applicable_templates


def test_get_template_known():
    t = get_template("unmeasured_confounding")
    assert t.kind == "confounding"
    assert "prevalence_u" in t.param_names()


def test_get_template_unknown_raises():
    with pytest.raises(KeyError):
        get_template("does_not_exist")


def test_validate_and_clip_clips_and_coerces():
    t = get_template("unmeasured_confounding")
    out = t.validate_and_clip(
        {"prevalence_u": 99, "rr_ud": "2.5", "rr_eu": 0.5}
    )
    assert out["prevalence_u"] == 0.6  # clipped high
    assert out["rr_ud"] == 2.5
    assert out["rr_eu"] == 0.9  # clipped low


def test_validate_and_clip_rejects_non_numeric():
    t = get_template("unmeasured_confounding")
    with pytest.raises(ValueError):
        t.validate_and_clip({"prevalence_u": "abc"})


def test_validate_and_clip_fills_defaults():
    t = get_template("unmeasured_confounding")
    out = t.validate_and_clip({})
    assert set(out) == set(t.param_names())


def test_applicable_templates():
    obs = applicable_templates("observational_cohort")
    ids = {t.id for t in obs}
    assert "unmeasured_confounding" in ids
    assert "differential_attrition" in ids
    rct = applicable_templates("randomized_trial")
    assert all("unmeasured_confounding" != t.id for t in rct)


def test_list_templates():
    assert len(list_templates()) == 2
