# AGENTS.md — Research Claim Stress-Tester

Guidance for agents working in this repository.

## Non-negotiable scope

- **Never overstate what a simulation proves.** A `POSSIBLE_UNDER_ASSUMPTIONS`
  verdict means the alternative is possible *under the declared assumptions*.
  It is never "the alternative is true," "the research is dishonest," or "the
  claim is false." Every evidence card must carry the disclaimer in
  `stresstest/evidence.py`.
- **Keep the proposer bounded.** The model may only fill parameters for the
  declared templates in `stresstest/templates.py`. Do not add free-form
  "alternative explanation" generation without a corresponding deterministic
  simulator and a clear verdict definition.
- **Deterministic code is the source of truth.** The simulator and discrimination
  check decide outcomes. The proposer only suggests.

## Meta-model / inference policy

- Live Meta-model inference is **disabled by default**. The free path uses
  **Meta open-weight (Llama) models** through a card-free partner host
  (Cloudflare Workers AI default; OVHcloud anonymous fallback). The newest
  Meta models (Muse Spark Open / Muse Glimmer) are **not** freely available
  without a card on any verified host — do not assume a free Muse path exists.
- The Meta Model API is **not** used: its $150 hackathon credits still require a
  payment method on file. `GuardedMetaProposer` is a disabled reference stub.
- `OpenAICompatibleMetaProposer` makes a live call **only** when explicitly
  enabled with a free token. On any failure it raises and the CLI falls back
  to the offline heuristic (labeled). Never fake model use in the demo.
- Do not add payment methods, submit provider forms, create provider accounts,
  enable auto-reload, or incur charges without explicit approval. Free tiers
  (no card) are acceptable; a spend *alert* is not a hard cap, but a free tier's
  daily quota (e.g. Cloudflare's 10,000 Neurons/day) is a hard cap enforced by
  the provider.
- Use synthetic inputs (we have none real). Cloudflare states it does not train
  on Workers AI customer content; verify the exact clause on their data page
  before the demo. Never put API keys, real user data, or secrets in the repo.

## Extending the MVP

- New rival templates require: a `RivalTemplate` spec, a simulator branch, a
  follow-up proposal, and a discrimination branch — plus tests. Do not add a
  template that has no deterministic check.
- Deferred modules (preregistration, multiverse exploration, plan-vs-actual
  auditing) stay out of scope unless justified for the demo.

## Compliance

- Hackathon build window and submission rules come from the **official** waiver
  (cerebralvalley.ai/e/meta-ai-developer-hackathon/waiver): judging
  Impact 30% / Meta AI tooling 25% / Technical depth 20% / Demo 25%; public repo
  + 3-min video required; new-work-only gate. The banned-project list is not in
  the public waiver; it was confirmed from the official Discord/portal and is
  treated as verified by the user.
- Only functionality built during the hackathon may be presented as new work.
  Do not present prior projects (e.g., GuardRail scaffold, Bendlab, sequential
  matching) as hackathon work or as demo fixtures.
- Solo; no teammates. No external/irreversible action (publishing, creating a
  remote repo, submitting) without explicit approval.

## Tests

```bash
python3 -m pytest -q
```

Do not claim tests pass unless they actually run here.
