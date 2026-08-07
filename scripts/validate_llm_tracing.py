#!/usr/bin/env python3
"""
HippocampAI: LLM Usage Tracing Bounded Validation Scenario
==============================================================
A self-contained, offline proof that HippocampAI's LLM tracing preserves
every upstream attempt behind a single logical model call, including a
provider fallback.

No network access. No API keys. No external routing layer (NovaRouteAI or
otherwise) is required or contacted. This validates the provider-neutral
trace *contract* only.

Scenario:
    Workflow "support-triage"
      -> Workflow Step "draft-reply"
        -> Agent "triage-agent"
          -> Logical LLM invocation (one client.llm.chat(...) call)
            -> Attempt 1: groq/llama-3.3-70b-versatile -> rate limited (retryable)
            -> Attempt 2: openai/gpt-4o-mini            -> provider fallback, succeeds

Run:
    python scripts/validate_llm_tracing.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hippocampai.telemetry import (  # noqa: E402
    UpstreamMetadata,
    get_telemetry,
    llm_attempt,
    llm_invocation,
    llm_trace_context,
)


def run_scenario() -> dict:
    telemetry = get_telemetry(enabled=True)

    with llm_trace_context(
        workflow_id="wf-support-triage",
        workflow_name="Support Triage",
        workflow_step_id="step-draft-reply",
        workflow_step_name="Draft Reply",
        agent_id="agent-triage",
        agent_name="Triage Agent",
        feature_name="draft_customer_reply",
        tenant_id="tenant-acme",
        workspace_id="workspace-support",
        user_id="user-agent-runner",
    ):
        with llm_invocation(
            provider="groq", requested_model="llama-3.3-70b-versatile", operation="chat"
        ) as invocation_id:
            # Attempt 1: primary provider is rate limited (retryable failure).
            # A router forwarding this request would tag it with its own
            # router_request_id - the same logical request, so the same ID
            # is expected to show up on every attempt underneath it. No
            # provider_request_id: the request never got a response.
            try:
                with llm_attempt(
                    provider="groq",
                    requested_model="llama-3.3-70b-versatile",
                    routing_reason="primary_provider",
                ) as attempt_1:
                    attempt_1.apply_metadata(
                        UpstreamMetadata(router_request_id="router-req-example-001")
                    )
                    raise TimeoutError(
                        "groq 429: rate limit exceeded, Authorization: Bearer gsk_should_never_appear_in_trace"
                    )
            except TimeoutError:
                pass  # caller (a router or this scenario) decides to fall back

            # Attempt 2: fall back to a different provider and model; succeeds.
            # This is a FALLBACK, not a retry of the same route - it counts
            # toward total_fallbacks, not total_retries (see validate()).
            with llm_attempt(
                provider="openai",
                requested_model="llama-3.3-70b-versatile",
                selected_model="gpt-4o-mini",
                fallback_reason="provider_rate_limited",
                routing_reason="secondary_provider",
            ) as attempt_2:
                # Simulate a normalized OpenAI-compatible response.
                fake_response = type(
                    "FakeResponse",
                    (),
                    {
                        "id": "chatcmpl-fallback-example",
                        "model": "gpt-4o-mini",
                        "usage": type(
                            "Usage",
                            (),
                            {
                                "prompt_tokens": 128,
                                "completion_tokens": 64,
                                "total_tokens": 192,
                                "prompt_tokens_details": None,
                                "completion_tokens_details": None,
                            },
                        )(),
                    },
                )()
                attempt_2.apply_metadata(UpstreamMetadata.from_openai_response(fake_response))
                # Same router_request_id as attempt 1: still the same logical
                # request as far as the router is concerned, even though it
                # was served by a different upstream provider.
                attempt_2.apply_metadata(
                    UpstreamMetadata(router_request_id="router-req-example-001")
                )
                attempt_2.finish_reason = "stop"

    invocation = telemetry.get_llm_invocation(invocation_id)
    assert invocation is not None
    return invocation


def validate(invocation) -> None:
    """Assert every invariant the tracing contract must preserve."""
    # Workflow / workflow-step identity is unchanged end to end.
    assert invocation.workflow_id == "wf-support-triage"
    assert invocation.workflow_name == "Support Triage"
    assert invocation.workflow_step_id == "step-draft-reply"
    assert invocation.workflow_step_name == "Draft Reply"
    assert invocation.agent_id == "agent-triage"
    assert invocation.agent_name == "Triage Agent"

    # Identity hierarchy: HippocampAI's own request_id, an external router's
    # own request_id, and each attempt's provider_request_id are three
    # distinct identities that are never conflated with one another.
    assert invocation.request_id
    assert invocation.trace_id
    assert invocation.router_request_id == "router-req-example-001"
    assert invocation.request_id != invocation.router_request_id

    # Retries are never collapsed: two distinct, ordered attempts exist.
    # This scenario is a pure fallback (attempt 2 switched provider/model
    # rather than retrying the same route), so total_retries is 0 and
    # total_fallbacks is 1 - a fallback is never double-counted as a retry.
    assert invocation.total_attempts == 2
    assert invocation.total_retries == 0
    assert invocation.total_fallbacks == 1
    attempt_ids = [a.attempt_id for a in invocation.attempts]
    assert len(set(attempt_ids)) == 2, "attempt IDs must be unique"
    assert [a.attempt_number for a in invocation.attempts] == [1, 2]

    attempt_1, attempt_2 = invocation.attempts

    # Attempt 1: requested vs selected model, retry/routing reason, per-attempt outcome.
    assert attempt_1.provider == "groq"
    assert attempt_1.requested_model == "llama-3.3-70b-versatile"
    assert attempt_1.status == "error"
    assert attempt_1.error_type == "timeout"
    # retry_reason belongs to whichever attempt was started BECAUSE this one
    # failed - not to the failing attempt itself (error_type already says
    # why it failed). Attempt 2 here is a fallback, not a same-route retry,
    # so retry_reason never gets inherited onto it either (see below).
    assert attempt_1.retry_reason is None
    assert attempt_1.routing_reason == "primary_provider"
    assert attempt_1.is_final is False
    assert attempt_1.latency_ms is not None and attempt_1.latency_ms >= 0
    assert attempt_1.router_request_id == "router-req-example-001"
    assert attempt_1.provider_request_id is None  # never got a response
    assert "gsk_should_never_appear_in_trace" not in (attempt_1.error_message or "")

    # Attempt 2: provider + model fallback, terminal success.
    assert attempt_2.provider == "openai"
    assert attempt_2.requested_model == "llama-3.3-70b-versatile"  # what was asked for
    assert attempt_2.selected_model == "gpt-4o-mini"  # what actually served it
    assert attempt_2.fallback_reason == "provider_rate_limited"
    assert attempt_2.routing_reason == "secondary_provider"
    assert attempt_2.retry_reason is None  # fallback, not a retry
    assert attempt_2.status == "success"
    assert attempt_2.is_final is True
    assert attempt_2.input_tokens == 128
    assert attempt_2.output_tokens == 64
    assert attempt_2.total_tokens == 192
    assert attempt_2.provider_request_id == "chatcmpl-fallback-example"
    assert attempt_2.router_request_id == "router-req-example-001"

    # Aggregation on the logical invocation reflects BOTH attempts, not just the winner.
    assert invocation.status == "success"
    assert invocation.selected_provider == "openai"
    assert invocation.selected_model == "gpt-4o-mini"
    assert invocation.input_tokens == 128
    assert invocation.output_tokens == 64
    assert invocation.total_tokens == 192
    assert invocation.latency_ms is not None
    assert invocation.latency_ms >= (attempt_1.latency_ms or 0) + (attempt_2.latency_ms or 0)
    # No pricing configured in this offline scenario -> cost is honestly unknown, not fabricated.
    assert invocation.estimated_cost is None


def to_sanitized_json(invocation) -> str:
    telemetry = get_telemetry()
    exported = telemetry.export_llm_invocations([invocation.invocation_id])[0]
    return json.dumps(exported, indent=2, sort_keys=True)


def main() -> int:
    invocation = run_scenario()
    validate(invocation)

    print("VALIDATION PASSED: all invariants preserved across retry + fallback.\n")
    print(
        f"total_attempts={invocation.total_attempts}  "
        f"total_retries={invocation.total_retries}  "
        f"total_fallbacks={invocation.total_fallbacks}"
    )
    print(
        f"final: {invocation.selected_provider}/{invocation.selected_model}  "
        f"status={invocation.status}  tokens={invocation.total_tokens}  "
        f"latency_ms={invocation.latency_ms:.2f}"
    )
    print("\nSanitized trace JSON:\n")
    print(to_sanitized_json(invocation))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
