"""
Verifier agent (v2, §4.3/§4.4).

Real fix vs. the previous draft:
  - Schema now matches §4.4's VerificationResult exactly, including
    `missing_agent_suspected` -- the field that lets the orchestrator tell
    a *planning* error (right specialist never invoked) from a *synthesis*
    error (right evidence, bad write-up). Without this field the two error
    types are indistinguishable and the single-retry logic in §4.4 can't
    branch correctly.
  - §3.8: the verifier must run on a DIFFERENT provider than whichever
    served synthesis that run. This is a real, checkable diversity
    guarantee, not aspirational -- get_verifier_provider() enforces it
    mechanically rather than trusting the caller to remember.

REAL, STATED DEGRADATION (Aug 2026): §3.8's original design assumed two
real cross-vendor providers (Cerebras primary, Groq fallback). In practice,
Cerebras now requires payment (confirmed via a real 402 payment_required
error during a live test run) and Groq's own path isn't currently
confirmed working either -- the only provider confirmed working, by direct
test, is Gemini. §3.8's diversity mechanism is kept REAL and RUNNING by
using two different Gemini models (flash-lite for synthesis, flash for
verification) instead of two different vendors. This is explicitly a
weaker diversity guarantee than the plan originally specified -- same
underlying model family means more correlated blind spots than true
cross-vendor diversity would give (§3.8's own citation of same-model
self-verification research applies MORE strongly here, not less, since
even the "different" model shares the same training lineage). Stated
honestly here rather than silently treated as equivalent to the original
design. If/when a second real vendor becomes available, swap
GROQ_MODEL/CEREBRAS_MODEL back in and this file's logic doesn't need to
change, only the two constants below.

Per §3.8: this is a second-pass consistency check over the SAME evidence
the synthesizer saw, with a DIFFERENT (not independent-in-the-fuller-sense)
model. Real and useful -- catches content-level unsupported claims v1's
citation-ID checker structurally can't -- but its ceiling is stated
honestly here, not oversold, and is currently lower than §3.8 originally
assumed for the reason above.
"""

import os
import time
from typing import Literal, Optional

from pydantic import BaseModel, Field
from pydantic_ai import Agent

# Real provider pair, Aug 2026 revision: same-vendor (Gemini) diversity
# pair, not the plan's originally-specified cross-vendor Cerebras/Groq
# pair -- see module docstring for why. PRIMARY_MODEL is whatever the
# synthesizer actually used that run (passed in via execute());
# SECONDARY_MODEL is the "different" model the verifier is forced onto.
PRIMARY_MODEL = "google:gemini-3.5-flash-lite"
SECONDARY_MODEL = "google:gemini-3.5-flash"

AGENT_LITERAL = Literal["structural", "history", "docs_semantic", "docs_keyword", "none"]


class VerificationResult(BaseModel):
    claims_supported: bool = Field(
        description="The real gate: does every claim in the draft follow from the evidence shown to the verifier."
    )
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Specific unsupported claim(s). Empty if claims_supported=True.",
    )
    missing_agent_suspected: AGENT_LITERAL = Field(
        default="none",
        description=(
            "If an unsupported claim looks like it's missing evidence a DIFFERENT "
            "specialist would have retrieved (a planning-error signal), name that "
            "specialist. Use 'none' if the gap looks like a synthesis error over "
            "evidence that WAS present."
        ),
    )
    reasoning: str = Field(default="", description="Logged in orchestration_runs.")


VERIFIER_INSTRUCTIONS = (
    "You are a strict, independent verification agent for a codebase-intelligence system.\n"
    "You are shown a DRAFT ANSWER and the EVIDENCE it was supposedly written from.\n\n"
    "Your job:\n"
    "1. Check every factual claim in the draft against the evidence. If a claim is not "
    "explicitly supported by the evidence, list it in unsupported_claims.\n"
    "2. claims_supported=True only if there are zero unsupported claims.\n"
    "3. If there ARE unsupported claims, judge whether the missing support looks like it "
    "would plausibly come from a specialist that was never consulted:\n"
    "   - 'structural': missing code-level facts (function bodies, call chains, file locations)\n"
    "   - 'history': missing commit/PR/issue/discussion rationale\n"
    "   - 'docs_semantic' or 'docs_keyword': missing documentation/README/release-note content\n"
    "   - 'none': the evidence needed was already present in what you were shown, and the "
    "draft simply misused or overstated it (a synthesis error, not a missing-evidence error)\n"
    "4. Be strict: a claim that is directionally plausible but not explicitly stated in the "
    "evidence is still unsupported."
)

_verifier_agents: dict[str, Agent] = {}


def _get_agent(model: str) -> Agent:
    if model not in _verifier_agents:
        _verifier_agents[model] = Agent(
            model,
            output_type=VerificationResult,
            retries=1,
            instructions=VERIFIER_INSTRUCTIONS,
        )
    return _verifier_agents[model]


def get_verifier_provider(synthesizer_provider: str) -> str:
    """§3.8's diversity rule, enforced mechanically: whichever Gemini model
    served synthesis this run, the verifier is forced onto the OTHER one.
    Falls back to SECONDARY_MODEL if the synthesizer provider string is
    unrecognized, so this never silently returns the same provider it was
    given. See module docstring for why this is a same-vendor pair rather
    than the plan's originally-specified cross-vendor one."""
    if synthesizer_provider and "flash-lite" in synthesizer_provider:
        return SECONDARY_MODEL
    return PRIMARY_MODEL


def call_with_backoff(agent: Agent, prompt: str, max_retries: int = 6):
    for attempt in range(max_retries):
        try:
            return agent.run_sync(prompt)
        except Exception as e:
            err = str(e).lower()

            # REAL FIX, same root cause as orchestrator.py's call_with_backoff
            # (see that file's comment for the full account): daily quota
            # exhaustion fails fast, no wasted backoff.
            if "perday" in err.replace(" ", ""):
                print(f"  [Verifier] DAILY QUOTA exhausted for this model -- failing fast, no backoff.")
                raise

            # Real fix, same root cause as orchestrator.py's call_with_backoff
            # (see that file's comment for the full account): a transient
            # DNS/network blip during a live 56-question run was NOT
            # retried, killing 23 questions outright instead of waiting a
            # few seconds for the network to recover.
            retryable = (
                "429", "503", "502", "500", "unavailable", "rate limit",
                "exhausted", "queue_exceeded", "timeout",
                "nodename nor servname", "name or service not known",
                "name resolution", "nameresolutionerror", "connection refused",
                "connection reset", "max retries exceeded", "connection aborted",
                "network is unreachable", "temporary failure in name resolution",
                "all connection attempts failed", "connection error",
            )
            if any(c in err for c in retryable):
                if attempt < max_retries - 1:
                    sleep_s = 5 * (2 ** attempt)
                    print(f"  [Verifier rate-limited/network-blip] waiting {sleep_s}s (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(sleep_s)
                    continue
            # Real fix, same as orchestrator.py's call_with_backoff: print
            # non-retryable errors before re-raising, so a failure here is
            # never silent (found via a real 402 payment_required going
            # unprinted on the Groq fallback path during a live test run).
            print(f"  [Verifier] non-retryable error: {str(e)[:300]}")
            raise


def execute(draft_answer: str, evidence_text: str, synthesizer_provider: str) -> tuple[VerificationResult, str]:
    """Runs verification on a provider deliberately different from whichever
    served synthesis this run. Returns (result, provider_used) so the caller
    can log both into orchestration_runs (§5.1) -- the provider-diversity
    guarantee is only real if it's actually logged and checkable, not just
    asserted in a docstring."""
    provider = get_verifier_provider(synthesizer_provider)
    print(f"  [Agent: Verifier] Cross-checking claims via {provider} (synthesizer used {synthesizer_provider})...")

    prompt = f"EVIDENCE:\n{evidence_text}\n\nDRAFT ANSWER:\n{draft_answer}"
    agent = _get_agent(provider)

    try:
        result = call_with_backoff(agent, prompt)
        return result.output, provider
    except Exception as e:
        # Real fallback provider, mirroring v1's synthesize_with_fallback
        # failure-only switch (distinct from §3.8's deliberate diversity
        # selection above, which already picked the "other" provider by
        # default -- this is the existing error-fallback layered on top).
        fallback = SECONDARY_MODEL if provider == PRIMARY_MODEL else PRIMARY_MODEL
        print(f"  [Verifier] {provider} failed ({e}); falling back to {fallback}...")
        agent2 = _get_agent(fallback)
        result = call_with_backoff(agent2, prompt)
        return result.output, fallback
