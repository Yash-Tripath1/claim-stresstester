"""Evidence-card rendering (Markdown + self-contained HTML).

The card always states: assumptions, simulation result, discrimination result,
limitations, and an explicit "what this does NOT prove" disclaimer.
"""

from __future__ import annotations

import html
from typing import List, Optional

from .claim import ResearchClaim
from .proposer import Proposal
from .simulator import SimulationResult
from .discrimination import FollowupProposal, DiscriminationResult

DISCLAIMER = (
    "This tool shows that an alternative explanation is POSSIBLE under explicitly "
    "stated modelling assumptions. It does NOT prove that the alternative caused "
    "the real result, does NOT detect dishonest research, and does NOT establish "
    "that the claim is false."
)


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.3g}"
    return str(v)


def render_markdown(
    claim: ResearchClaim,
    proposal: Proposal,
    sim: SimulationResult,
    followup: Optional[FollowupProposal],
    disc: Optional[DiscriminationResult],
) -> str:
    lines: List[str] = []
    lines.append("# Evidence Card — Research Claim Stress-Test\n")
    lines.append(f"**Study type:** {claim.study_type}  ")
    lines.append(
        f"**Reported effect:** {claim.effect_estimate} {claim.outcome_unit} "
        f"(p = {claim.p_value:.3g}, N ≈ {claim.n_per_arm}/arm)"
    )
    if claim.effect_ci:
        lines.append(
            f"**Reported 95% CI:** {claim.effect_ci[0]} to {claim.effect_ci[1]}"
        )
    lines.append(f"**Baseline / SD:** {claim.baseline_outcome} / {claim.outcome_sd}\n")

    lines.append("## Alternative explanation considered\n")
    lines.append(f"**Template:** {proposal.template_id}  ")
    lines.append(f"**Proposer source:** {proposal.source}\n")
    lines.append("**Proposed parameters:**\n")
    for k, v in proposal.parameters.items():
        lines.append(f"- `{k}` = {_fmt(v)}")
    lines.append(f"\n**Rationale:** {proposal.rationale}\n")

    lines.append("## Simulation result\n")
    lines.append(f"**Verdict:** `{sim.verdict}`\n")
    lines.append(
        f"- Apparent effect (mean): **{sim.apparent_effect_mean:.2f}** "
        f"{claim.outcome_unit} "
        f"(95% CI {sim.apparent_effect_ci[0]:.2f} to {sim.apparent_effect_ci[1]:.2f})"
    )
    lines.append(
        f"- Fraction of simulations reproducing the claim: "
        f"**{sim.fraction_matching_claim:.1%}**"
    )
    lines.append(f"- Replicates: {sim.n_replicates} (seed {sim.seed})\n")
    lines.append("**Assumptions:**\n")
    for k, v in sim.assumptions.items():
        lines.append(f"- `{k}` = {_fmt(v)}")
    lines.append(f"\n**Note:** {sim.notes}\n")

    if followup and disc:
        lines.append("## Follow-up and discrimination check\n")
        lines.append(f"**Follow-up design:** {followup.design}\n")
        lines.append(f"**Distinguishing test:** {followup.test} (N = {followup.sample_size})\n")
        lines.append(f"**Discrimination verdict:** `{disc.verdict}`\n")
        lines.append(
            f"- Combined power: **{disc.power:.1%}** "
            f"(World A {disc.power_world_A:.1%}, World B {disc.power_world_B:.1%})"
        )
        lines.append(
            f"- Expected effect — alternative world: {disc.expected_effect_world_A:.2f}; "
            f"claim world: {disc.expected_effect_world_B:.2f}"
        )
        lines.append(f"- SE ≈ {disc.se:.2f}; N_follow = {disc.n_follow}")
        lines.append(f"\n**Note:** {disc.notes}\n")
    else:
        lines.append("## Follow-up and discrimination check\n")
        lines.append(
            "_Not run: the alternative was not found possible under the stated "
            "assumptions, so a discriminating follow-up was not proposed._\n"
        )

    lines.append("## Limitations\n")
    lines.append(
        "- The simulation is a synthetic generative model; it reflects the "
        "assumptions declared above, not the real study."
    )
    lines.append(
        "- Only the declared parameters were varied; other explanations were "
        "not considered."
    )
    lines.append(
        "- The proposer in this demo is a deterministic offline heuristic, not a "
        "verified hosted Meta model."
    )
    lines.append("")
    lines.append("## What this does NOT prove\n")
    lines.append(f"> {DISCLAIMER}\n")
    return "\n".join(lines)


def render_html(
    claim: ResearchClaim,
    proposal: Proposal,
    sim: SimulationResult,
    followup: Optional[FollowupProposal],
    disc: Optional[DiscriminationResult],
) -> str:
    def esc(s) -> str:
        return html.escape(str(s))

    verdict_color = {
        "POSSIBLE_UNDER_ASSUMPTIONS": "#b26a00",
        "REJECTED_UNDER_ASSUMPTIONS": "#1b7f3b",
        "UNCERTAIN": "#6b6b6b",
        "DISCRIMINATES": "#1b7f3b",
        "WEAK": "#b26a00",
        "DOES_NOT_DISCRIMINATE": "#a00",
    }.get(sim.verdict, "#333")

    disc_html = ""
    if followup and disc:
        dv = disc.verdict
        dc = {
            "DISCRIMINATES": "#1b7f3b",
            "WEAK": "#b26a00",
            "DOES_NOT_DISCRIMINATE": "#a00",
        }.get(dv, "#333")
        disc_html = f"""
        <section>
          <h2>Follow-up and discrimination check</h2>
          <p>{esc(followup.design)}</p>
          <p><b>Distinguishing test:</b> {esc(followup.test)} (N = {followup.sample_size})</p>
          <p><span class="badge" style="background:{dc};color:#fff">{esc(dv)}</span></p>
          <ul>
            <li>Combined power: <b>{disc.power:.1%}</b> (World A {disc.power_world_A:.1%}, World B {disc.power_world_B:.1%})</li>
            <li>Expected effect — alternative world: {disc.expected_effect_world_A:.2f}; claim world: {disc.expected_effect_world_B:.2f}</li>
            <li>SE ≈ {disc.se:.2f}; N_follow = {disc.n_follow}</li>
          </ul>
          <p><em>{esc(disc.notes)}</em></p>
        </section>"""

    params_html = "".join(
        f"<li><code>{esc(k)}</code> = {esc(_fmt(v))}</li>" for k, v in proposal.parameters.items()
    )
    assumptions_html = "".join(
        f"<li><code>{esc(k)}</code> = {esc(_fmt(v))}</li>" for k, v in sim.assumptions.items()
    )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Evidence Card — Research Claim Stress-Test</title>
<style>
  body{{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
       max-width:760px;margin:2rem auto;padding:0 1rem;color:#1a1a1a;line-height:1.5}}
  h1{{font-size:1.5rem}} h2{{font-size:1.15rem;margin-top:1.6rem;border-bottom:1px solid #e2e2e2;padding-bottom:.2rem}}
  code{{background:#f3f3f3;padding:.1rem .3rem;border-radius:4px;font-size:.9em}}
  .badge{{display:inline-block;padding:.2rem .6rem;border-radius:6px;font-weight:600;font-size:.85rem}}
  .disclaimer{{background:#fff7e6;border:1px solid #f0c36d;border-radius:8px;padding:.8rem 1rem;margin-top:1rem}}
  ul{{padding-left:1.2rem}} li{{margin:.2rem 0}}
  .muted{{color:#666;font-size:.9rem}}
</style></head><body>
  <h1>Evidence Card — Research Claim Stress-Test</h1>
  <p class="muted">Offline deterministic simulation · synthetic inputs</p>
  <section>
    <h2>Claim</h2>
    <ul>
      <li>Study type: {esc(claim.study_type)}</li>
      <li>Reported effect: <b>{claim.effect_estimate} {esc(claim.outcome_unit)}</b>
          (p = {claim.p_value:.3g}, N ≈ {claim.n_per_arm}/arm)</li>
      <li>Baseline / SD: {claim.baseline_outcome} / {claim.outcome_sd}</li>
    </ul>
  </section>
  <section>
    <h2>Alternative explanation considered</h2>
    <p><b>Template:</b> {esc(proposal.template_id)} ·
       <b>Proposer:</b> {esc(proposal.source)}</p>
    <ul>{params_html}</ul>
    <p><em>{esc(proposal.rationale)}</em></p>
  </section>
  <section>
    <h2>Simulation result</h2>
    <p><span class="badge" style="background:{verdict_color};color:#fff">{esc(sim.verdict)}</span></p>
    <ul>
      <li>Apparent effect (mean): <b>{sim.apparent_effect_mean:.2f}</b> {esc(claim.outcome_unit)}
          (95% CI {sim.apparent_effect_ci[0]:.2f} to {sim.apparent_effect_ci[1]:.2f})</li>
      <li>Fraction of simulations reproducing the claim: <b>{sim.fraction_matching_claim:.1%}</b></li>
      <li>Replicates: {sim.n_replicates} (seed {sim.seed})</li>
    </ul>
    <p><b>Assumptions:</b></p>
    <ul>{assumptions_html}</ul>
    <p><em>{esc(sim.notes)}</em></p>
  </section>
  {disc_html}
  <section>
    <h2>Limitations</h2>
    <ul>
      <li>The simulation is a synthetic generative model reflecting the declared assumptions, not the real study.</li>
      <li>Only the declared parameters were varied; other explanations were not considered.</li>
      <li>The proposer in this demo is a deterministic offline heuristic, not a verified hosted Meta model.</li>
    </ul>
  </section>
  <section>
    <h2>What this does NOT prove</h2>
    <div class="disclaimer">{esc(DISCLAIMER)}</div>
  </section>
</body></html>"""
