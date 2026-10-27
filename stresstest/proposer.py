"""Rival-explanation proposers.

Three proposers ship in this MVP:

* ``DeterministicFallbackProposer`` -- a transparent, offline heuristic that
  proposes *candidate* parameters for a template. It is NOT a neural model and
  makes no claims about being one. It is the default so the demo runs with no
  provider and no card.

* ``OpenAICompatibleMetaProposer`` -- proposes parameters using a hosted Meta
  open-weight model (Llama) through any OpenAI-compatible ``/chat/completions``
  endpoint (e.g. Cloudflare Workers AI, OVHcloud AI Endpoints). Both free tiers
  require **no credit card**. The proposer is DISABLED by default and makes no
  live call unless explicitly enabled with a token. On any failure it raises,
  and the CLI falls back to the deterministic proposer (clearly labelled).

* ``GuardedMetaProposer`` -- legacy stub for the Meta Model API (requires a
  card). Kept for reference; disabled and not used.

The proposer only *suggests* parameters. The deterministic simulator decides
whether the suggestion actually reproduces the reported result.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Dict, Optional

from .claim import ResearchClaim
from .templates import RivalTemplate


@dataclass
class Proposal:
    template_id: str
    parameters: Dict[str, float]
    rationale: str
    source: str  # "offline_deterministic_fallback" | "meta_model_open_weight" | "meta_model"


class DeterministicFallbackProposer:
    """Offline heuristic proposer. Deterministic and auditable."""

    source = "offline_deterministic_fallback"

    def propose(self, claim: ResearchClaim, template: RivalTemplate) -> Proposal:
        if template.id == "unmeasured_confounding":
            params = {
                "prevalence_u": 0.30,
                "rr_ud": 3.0,
                "rr_eu": 1.22,
            }
            rationale = (
                "Heuristic candidate: a moderately common confounder (30% of the "
                "sample) that strongly predicts treatment receipt (RR~3) and shifts "
                "the conditional mean outcome (~+22%) can, under a zero true effect, "
                "manufacture an apparent effect of roughly the reported magnitude."
            )
        elif template.id == "differential_attrition":
            params = {
                "attrition_rate_trt": 0.30,
                "attrition_rate_ctrl": 0.10,
                "attrite_outcome_trt_offset": -12.0,
                "attrite_outcome_ctrl_offset": 0.0,
            }
            rationale = (
                "Heuristic candidate: if 30% of the treatment arm is lost and those "
                "dropouts score ~12 points below baseline, while the control arm loses "
                "a representative 10%, the surviving treatment completers look ~5 "
                "points better even with a zero true effect."
            )
        else:
            raise ValueError(f"No fallback heuristic for template '{template.id}'")

        clipped = template.validate_and_clip(params)
        return Proposal(
            template_id=template.id,
            parameters=clipped,
            rationale=rationale,
            source=self.source,
        )


def _extract_content(payload: dict) -> str:
    """Pull assistant text out of OpenAI or Cloudflare Workers AI response shapes."""
    # OpenAI: choices[0].message.content
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        msg = choices[0].get("message", {})
        content = msg.get("content")
        if isinstance(content, str):
            return content
    # Cloudflare Workers AI: result.message.content (string or list of blocks)
    result = payload.get("result")
    if isinstance(result, dict):
        msg = result.get("message", {})
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    parts.append(block["text"])
            return "".join(parts)
    raise ValueError(f"Could not extract content from response: {payload!r}")


class OpenAICompatibleMetaProposer:
    """Proposes parameters with a hosted Meta open-weight (Llama) model.

    Works with any OpenAI-compatible ``/chat/completions`` endpoint, e.g.:

    * Cloudflare Workers AI (free tier, no card):
      base ``https://api.cloudflare.com/client/v4/ai/run``,
      model ``@cf/meta/llama-3.3-70b-instruct-fp8-fast``,
      ``format="cloudflare"``.
    * OVHcloud AI Endpoints (anonymous, no card):
      base ``https://oai.endpoints.kepler.ai.cloud.ovh.net/v1``,
      model ``Meta-Llama-3_3-70B-Instruct``, ``format="openai"``.

    Disabled unless ``enabled=True`` and ``api_key`` is set. No live call is
    made otherwise. A ``transport`` callable can be injected for testing.
    """

    source = "meta_model_open_weight"

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.cloudflare.com/client/v4/ai/run",
        model: str = "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
        format: str = "cloudflare",
        enabled: bool = False,
        timeout: float = 30.0,
        max_tokens: int = 400,
        transport: Optional[Callable[[str, dict, dict, float], dict]] = None,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.format = format
        self.enabled = bool(enabled and api_key)
        self.timeout = timeout
        self.max_tokens = max_tokens
        self._transport = transport

    def is_live(self) -> bool:
        return self.enabled

    # -- public API --------------------------------------------------------
    def propose(self, claim: ResearchClaim, template: RivalTemplate) -> Proposal:
        if not self.enabled:
            raise RuntimeError(
                "OpenAI-compatible Meta proposer is disabled (no api_key). "
                "No live call is made."
            )
        prompt = self._build_prompt(claim, template)
        text = self._chat(prompt)
        data = self._parse_json(text)
        if "parameters" not in data:
            raise ValueError(f"Model response missing 'parameters': {data!r}")
        params = template.validate_and_clip(data["parameters"])
        rationale = str(data.get("rationale", "")).strip()
        return Proposal(template.id, params, rationale, self.source)

    # -- internals ---------------------------------------------------------
    def _build_prompt(self, claim: ResearchClaim, template: RivalTemplate) -> str:
        param_lines = []
        for spec in template.params:
            param_lines.append(
                f"- {spec.name} ({spec.type}, range [{spec.low}, {spec.high}]): "
                f"{spec.description}"
            )
        return (
            "You are a careful scientific critic. Given a research claim and a "
            "bounded alternative-explanation template, propose plausible values "
            "for the template's parameters that could, under a zero true effect, "
            "reproduce the reported result. Respond ONLY with strict JSON of the "
            'form {"parameters": {...}, "rationale": "..."} with no extra text.\n\n'
            f"CLAIM: {claim.raw_text}\n"
            f"Study type: {claim.study_type}; effect: {claim.effect_estimate} "
            f"{claim.outcome_unit}; p: {claim.p_value}; N/arm: {claim.n_per_arm}; "
            f"baseline: {claim.baseline_outcome}; SD: {claim.outcome_sd}.\n\n"
            f"TEMPLATE: {template.name} ({template.id})\n"
            f"{template.description}\n\n"
            "Parameters to fill:\n" + "\n".join(param_lines)
        )

    def _chat(self, prompt: str) -> str:
        if self.format == "cloudflare":
            url = self.base_url  # e.g. .../ai/run
        else:  # openai-compatible
            url = f"{self.base_url}/chat/completions"
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": self.max_tokens,
        }
        payload = self._post(url, body)
        return _extract_content(payload)

    def _post(self, url: str, body: dict) -> dict:
        if self._transport is not None:
            return self._transport(url, body, self._headers(), self.timeout)
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers=self._headers(),
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:200]
            raise RuntimeError(f"Meta proposer HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Meta proposer network error: {exc}") from exc

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _parse_json(text: str) -> dict:
        # Strip code fences if the model wrapped the JSON.
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip().rstrip("`").strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            # Try to find the first {...} block.
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(cleaned[start : end + 1])
                except json.JSONDecodeError:
                    pass
            raise ValueError(f"Could not parse JSON from model: {text!r}") from exc


class GuardedMetaProposer:
    """Legacy stub for the Meta Model API (requires a card). DISABLED.

    Kept for reference. The Meta Model API credits require a payment method on
    file, so this path is not used. Live inference is intentionally not
    implemented.
    """

    source = "meta_model"

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        enabled: bool = False,
        allow_live: bool = False,
    ):
        self.provider = provider
        self.model = model
        self.enabled = enabled
        self.allow_live = allow_live

    @property
    def is_live(self) -> bool:
        return bool(self.enabled and self.allow_live and self.provider and self.model)

    def propose(self, claim: ResearchClaim, template: RivalTemplate) -> Proposal:
        if not self.is_live:
            raise RuntimeError(
                "Live Meta-model inference (Meta Model API) is disabled and "
                "requires a card. No provider call is made."
            )
        raise NotImplementedError(
            "Meta Model API live path is intentionally unimplemented."
        )


def default_proposer() -> DeterministicFallbackProposer:
    return DeterministicFallbackProposer()
