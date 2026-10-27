import pytest

from stresstest.claim import load_claim
from stresstest.templates import get_template
from stresstest.proposer import (
    DeterministicFallbackProposer,
    OpenAICompatibleMetaProposer,
    _extract_content,
)
from stresstest.cli import _make_proposer, _propose_with_fallback

FIXTURE = "fixtures/tutoring_claim.json"


@pytest.fixture
def claim():
    return load_claim(FIXTURE)


def _confounding(claim):
    return get_template("unmeasured_confounding")


# -- disabled by default ---------------------------------------------------
def test_disabled_without_api_key_raises(claim):
    t = _confounding(claim)
    p = OpenAICompatibleMetaProposer(api_key=None, enabled=False)
    assert p.is_live() is False
    with pytest.raises(RuntimeError):
        p.propose(claim, t)


# -- cloudflare response shape -------------------------------------------
def test_cloudflare_response_parsed(claim):
    t = _confounding(claim)

    def transport(url, body, headers, timeout):
        assert body["model"] == "@cf/meta/llama-3.3-70b-instruct-fp8-fast"
        assert headers["Authorization"] == "Bearer k"
        return {
            "result": {
                "message": {
                    "content": '{"parameters": {"prevalence_u": 0.25, "rr_ud": 2.5, "rr_eu": 1.2}, "rationale": "cf-test"}'
                }
            }
        }

    p = OpenAICompatibleMetaProposer(
        api_key="k", enabled=True, transport=transport
    )
    prop = p.propose(claim, t)
    assert prop.parameters == {"prevalence_u": 0.25, "rr_ud": 2.5, "rr_eu": 1.2}
    assert prop.rationale == "cf-test"
    assert prop.source == "meta_model_open_weight"


# -- openai-compatible (OVH) response shape ------------------------------
def test_openai_response_parsed(claim):
    t = _confounding(claim)

    def transport(url, body, headers, timeout):
        assert url.endswith("/chat/completions")
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"parameters": {"prevalence_u": 0.2, "rr_ud": 2.0, "rr_eu": 1.15}, "rationale": "ovh-test"}'
                    }
                }
            ]
        }

    p = OpenAICompatibleMetaProposer(
        api_key="k",
        base_url="https://oai.endpoints.kepler.ai.cloud.ovh.net/v1",
        model="Meta-Llama-3_3-70B-Instruct",
        format="openai",
        enabled=True,
        transport=transport,
    )
    prop = p.propose(claim, t)
    assert prop.parameters["prevalence_u"] == 0.2
    assert prop.source == "meta_model_open_weight"


# -- bad JSON raises ------------------------------------------------------
def test_bad_json_raises(claim):
    t = _confounding(claim)

    def transport(url, body, headers, timeout):
        return {"result": {"message": {"content": "not json at all"}}}

    p = OpenAICompatibleMetaProposer(api_key="k", enabled=True, transport=transport)
    with pytest.raises(ValueError):
        p.propose(claim, t)


# -- content extraction shapes -------------------------------------------
def test_extract_content_openai():
    payload = {"choices": [{"message": {"content": "hello"}}]}
    assert _extract_content(payload) == "hello"


def test_extract_content_cloudflare_string():
    payload = {"result": {"message": {"content": "hello"}}}
    assert _extract_content(payload) == "hello"


def test_extract_content_cloudflare_blocks():
    payload = {
        "result": {
            "message": {
                "content": [
                    {"type": "text", "text": "hello "},
                    {"type": "text", "text": "world"},
                ]
            }
        }
    }
    assert _extract_content(payload) == "hello world"


# -- parameters clipped to template bounds ------------------------------
def test_params_clipped_to_bounds(claim):
    t = _confounding(claim)

    def transport(url, body, headers, timeout):
        return {
            "result": {
                "message": {
                    "content": '{"parameters": {"prevalence_u": 99, "rr_ud": 0.1, "rr_eu": 1.2}, "rationale": "x"}'
                }
            }
        }

    p = OpenAICompatibleMetaProposer(api_key="k", enabled=True, transport=transport)
    prop = p.propose(claim, t)
    assert prop.parameters["prevalence_u"] == 0.6  # clipped high
    assert prop.parameters["rr_ud"] == 0.5  # clipped low


# -- CLI proposer factory -------------------------------------------------
def test_make_proposer_offline_by_default(monkeypatch):
    monkeypatch.delenv("STRESSTEST_API_KEY", raising=False)
    args = type("A", (), {"proposer": "cloudflare"})()
    p = _make_proposer(args)
    assert isinstance(p, DeterministicFallbackProposer)


def test_make_proposer_live_when_key_present(monkeypatch):
    monkeypatch.setenv("STRESSTEST_API_KEY", "k")
    args = type("A", (), {"proposer": "cloudflare"})()
    p = _make_proposer(args)
    assert isinstance(p, OpenAICompatibleMetaProposer)
    assert p.is_live() is True


# -- CLI fallback when live proposer fails ------------------------------
def test_propose_with_fallback_on_error(claim):
    t = _confounding(claim)

    class _Broken:
        source = "meta_model_open_weight"

        def propose(self, claim, template):
            raise RuntimeError("boom")

    offline = DeterministicFallbackProposer()
    prop = _propose_with_fallback(claim, t, _Broken(), offline)
    assert prop.source == "offline_deterministic_fallback"
    assert "live proposer unavailable" in prop.rationale


def test_propose_with_fallback_uses_offline_when_primary_is_offline(claim):
    t = _confounding(claim)
    offline = DeterministicFallbackProposer()
    prop = _propose_with_fallback(claim, t, offline, offline)
    assert prop.source == "offline_deterministic_fallback"
