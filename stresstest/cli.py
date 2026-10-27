"""Command-line demo.

Runs the full offline stress-test flow on a synthetic claim and writes an
evidence card (Markdown + HTML) and a JSON report.

    python -m stresstest demo [--out OUT_DIR]
    python -m stresstest check-claim TEXT [--out OUT_DIR]

Live Meta-model inference is NOT used: the demo runs the deterministic offline
fallback proposer and a seeded Monte-Carlo simulator.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

from .claim import ResearchClaim, load_claim, parse_claim_text
from .templates import get_template, list_templates, applicable_templates
from .proposer import (
    DeterministicFallbackProposer,
    GuardedMetaProposer,
    OpenAICompatibleMetaProposer,
    Proposal,
)
from .simulator import Simulator, POSSIBLE
from .discrimination import propose_followup, power_check
from .evidence import render_markdown, render_html

import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FIXTURE = PROJECT_ROOT / "fixtures" / "tutoring_claim.json"


def _make_proposer(args) -> object:
    """Build the active proposer from CLI args / env, offline by default."""
    mode = (args.proposer or os.environ.get("STRESSTEST_PROPOSER") or "offline").lower()
    api_key = os.environ.get("STRESSTEST_API_KEY")
    if mode in ("cloudflare", "openai") and api_key:
        fmt = "cloudflare" if mode == "cloudflare" else "openai"
        base = os.environ.get(
            "STRESSTEST_BASE_URL",
            "https://api.cloudflare.com/client/v4/ai/run" if fmt == "cloudflare"
            else "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1",
        )
        model = os.environ.get(
            "STRESSTEST_MODEL",
            "@cf/meta/llama-3.3-70b-instruct-fp8-fast" if fmt == "cloudflare"
            else "Meta-Llama-3_3-70B-Instruct",
        )
        return OpenAICompatibleMetaProposer(
            api_key=api_key, base_url=base, model=model, format=fmt, enabled=True
        )
    return DeterministicFallbackProposer()


def _propose_with_fallback(claim, template, primary, offline) -> Proposal:
    """Try the live proposer; on any failure, fall back to the offline heuristic."""
    if isinstance(primary, DeterministicFallbackProposer):
        return primary.propose(claim, template)
    try:
        return primary.propose(claim, template)
    except Exception as exc:  # noqa: BLE001 - deliberate broad fallback
        fallback = offline.propose(claim, template)
        fallback.rationale = f"[live proposer unavailable: {exc}] {fallback.rationale}"
        return fallback


def _run_one(
    claim: ResearchClaim,
    template_id: str,
    proposer,
    simulator: Simulator,
) -> dict:
    template = get_template(template_id)
    offline = DeterministicFallbackProposer()
    proposal = _propose_with_fallback(claim, template, proposer, offline)
    sim = simulator.evaluate(claim, template, proposal.parameters)
    followup = None
    disc = None
    if sim.verdict == POSSIBLE:
        followup = propose_followup(claim, template, proposal.parameters)
        disc = power_check(claim, template, proposal.parameters, followup)

    return {
        "template_id": template_id,
        "template_name": template.name,
        "proposal": proposal,
        "simulation": sim,
        "followup": followup,
        "discrimination": disc,
    }


def run_demo(
    out_dir: Path,
    claim: ResearchClaim,
    n_replicates: int = 1000,
    proposer_mode: str = "offline",
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    args = type("A", (), {"proposer": proposer_mode})()
    proposer = _make_proposer(args)
    live_mode = "meta_model_open_weight" if isinstance(
        proposer, OpenAICompatibleMetaProposer
    ) else "deterministic_fallback"
    simulator = Simulator(n_replicates=n_replicates, seed=12345)

    applicable = applicable_templates(claim.study_type)
    if not applicable:
        applicable = list_templates()

    results = [_run_one(claim, t.id, proposer, simulator) for t in applicable]

    # Evidence cards per alternative + a combined HTML report.
    cards_md: List[str] = []
    cards_html: List[str] = []
    for r in results:
        md = render_markdown(claim, r["proposal"], r["simulation"], r["followup"], r["discrimination"])
        html = render_html(claim, r["proposal"], r["simulation"], r["followup"], r["discrimination"])
        safe = r["template_id"]
        (out_dir / f"evidence_{safe}.md").write_text(md)
        (out_dir / f"evidence_{safe}.html").write_text(html)
        cards_md.append(md)
        cards_html.append(html)

    combined_html = (
        "<html><head><meta charset='utf-8'>"
        "<style>body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,"
        "sans-serif;max-width:820px;margin:2rem auto;padding:0 1rem;color:#1a1a1a;"
        "line-height:1.5}h1{font-size:1.5rem}</style></head><body>"
        "<h1>Research Claim Stress-Test — Evidence Report</h1>"
        "<p>Offline deterministic simulation · synthetic inputs · "
        "POSSIBLE means possible under assumptions, not proven.</p>"
        + "".join(cards_html)
        + "</body></html>"
    )
    (out_dir / "evidence_report.html").write_text(combined_html)

    report = {
        "claim": claim.to_dict(),
        "offline": live_mode == "deterministic_fallback",
        "proposer": live_mode,
        "results": [
            {
                "template_id": r["template_id"],
                "template_name": r["template_name"],
                "proposal": {
                    "source": r["proposal"].source,
                    "rationale": r["proposal"].rationale,
                    "parameters": r["proposal"].parameters,
                },
                "simulation": r["simulation"].to_dict(),
                "followup": r["followup"].to_dict() if r["followup"] else None,
                "discrimination": r["discrimination"].to_dict() if r["discrimination"] else None,
            }
            for r in results
        ],
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
    return report


def _print_summary(report: dict) -> None:
    claim = report["claim"]
    print("=" * 60)
    print("RESEARCH CLAIM STRESS-TEST (offline demo)")
    print("=" * 60)
    print(
        f"Claim: {claim['effect_estimate']} {claim['outcome_unit']} "
        f"(p={claim['p_value']:.3g}, N~{claim['n_per_arm']}/arm, "
        f"{claim['study_type']})"
    )
    print(f"Proposer: {report['proposer']}  (live Meta inference: disabled)")
    print("-" * 60)
    for r in report["results"]:
        sim = r["simulation"]
        print(f"[{r['template_name']}] -> {sim['verdict']}")
        print(
            f"   apparent effect mean = {sim['apparent_effect_mean']:.2f} "
            f"(95% CI {sim['apparent_effect_ci'][0]:.2f}..{sim['apparent_effect_ci'][1]:.2f}); "
            f"match frac = {sim['fraction_matching_claim']:.1%}"
        )
        if r["discrimination"]:
            d = r["discrimination"]
            print(
                f"   follow-up: {r['followup']['test']} -> {d['verdict']} "
                f"(power {d['power']:.1%})"
            )
        else:
            print("   follow-up: not proposed (alternative not possible under assumptions)")
    print("-" * 60)
    print("A POSSIBLE verdict means the alternative is possible under the stated")
    print("assumptions -- it does NOT prove the alternative, nor disprove the claim.")
    print("=" * 60)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="stresstest")
    parser.add_argument("command", choices=["demo", "check-claim"])
    parser.add_argument("--out", default="out", help="output directory")
    parser.add_argument("--fixture", default=str(DEFAULT_FIXTURE))
    parser.add_argument("--text", default=None, help="claim text (for check-claim)")
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument(
        "--proposer",
        choices=["offline", "cloudflare", "openai"],
        default=None,
        help="proposer mode (default: env STRESSTEST_PROPOSER or offline)",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out)

    if args.command == "demo":
        claim = load_claim(args.fixture)
    else:
        if not args.text:
            parser.error("check-claim requires --text")
        claim = parse_claim_text(args.text)

    report = run_demo(
        out_dir, claim, n_replicates=args.replicates, proposer_mode=args.proposer or "offline"
    )
    _print_summary(report)
    print(f"\nWrote evidence cards + report to: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
