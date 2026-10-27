"""Research Claim Stress-Tester.

A small, offline toolkit that stress-tests a research claim against a *bounded*
set of rival explanations. The model's role is deliberately narrow: it proposes
parameters for one of a fixed set of alternative-explanation templates.
Deterministic, seeded Monte-Carlo code then checks whether that alternative
could actually produce the reported result under explicitly stated assumptions,
and whether a proposed follow-up could distinguish the explanations.

IMPORTANT SCOPE GUARD
----------------------
A "POSSIBLE_UNDER_ASSUMPTIONS" verdict means: under the declared assumptions,
this alternative explanation is *able to reproduce* the apparent result. It does
NOT mean the alternative explanation is true, that it caused the real result,
that the research is dishonest, or that the claim is false. The simulator
never makes those judgments.
"""

from .claim import ResearchClaim, load_claim, parse_claim_text
from .templates import (
    RivalTemplate,
    ParamSpec,
    get_template,
    applicable_templates,
    list_templates,
)
from .proposer import (
    Proposal,
    DeterministicFallbackProposer,
    GuardedMetaProposer,
)
from .simulator import Simulator, SimulationResult
from .discrimination import (
    FollowupProposal,
    DiscriminationResult,
    propose_followup,
    power_check,
)

__version__ = "0.1.0"

__all__ = [
    "ResearchClaim",
    "load_claim",
    "parse_claim_text",
    "RivalTemplate",
    "ParamSpec",
    "get_template",
    "applicable_templates",
    "list_templates",
    "Proposal",
    "DeterministicFallbackProposer",
    "GuardedMetaProposer",
    "Simulator",
    "SimulationResult",
    "FollowupProposal",
    "DiscriminationResult",
    "propose_followup",
    "power_check",
    "__version__",
]
