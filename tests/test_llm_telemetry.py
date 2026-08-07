"""Tests for LLM usage tracing: logical invocations and upstream attempts.

Covers the two-level trace model added on top of ``hippocampai.telemetry``:
one ``LLMInvocation`` (the logical model call an application made) can
aggregate several ``UpstreamAttempt`` records (the real provider requests
behind it - retries, rate limits, timeouts, fallback).

All provider calls are mocked; no network access is used.
"""

from __future__ import annotations

import asyncio
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

import hippocampai.telemetry as telemetry_module
from hippocampai.telemetry import (
    LLMTraceContext,
    MemoryTelemetry,
    UpstreamMetadata,
    _cap_metadata,
    classify_llm_error,
    estimate_cost,
    get_telemetry,
    llm_attempt,
    llm_invocation,
    llm_trace_context,
    sanitize_error_message,
)


@pytest.fixture(autouse=True)
def _no_real_retry_backoff(monkeypatch):
    """Tenacity's exponential backoff really sleeps between retries; these
    tests only care about attempt bookkeeping, not wall-clock backoff."""
    monkeypatch.setattr(time, "sleep", lambda seconds: None)


@pytest.fixture
def telemetry():
    """A fresh, enabled global telemetry instance for each test."""
    telemetry_module._global_telemetry = None
    instance = get_telemetry(enabled=True)
    yield instance
    telemetry_module._global_telemetry = None


@pytest.fixture
def openai_llm():
    from hippocampai.adapters.provider_openai import OpenAILLM

    llm = OpenAILLM.__new__(OpenAILLM)
    llm.model = "gpt-4o-mini"
    llm.client = MagicMock()
    return llm


def _openai_response(
    content: str = "hello",
    request_id: str = "req_123",
    model: str = "gpt-4o-mini",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    finish_reason: str = "stop",
):
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.total_tokens = prompt_tokens + completion_tokens
    usage.prompt_tokens_details = None
    usage.completion_tokens_details = None

    choice = MagicMock()
    choice.finish_reason = finish_reason
    choice.message.content = content

    response = MagicMock()
    response.id = request_id
    response.model = model
    response.usage = usage
    response.choices = [choice]
    return response


# ---------------------------------------------------------------------
# 1. Single successful logical call, one upstream attempt
# ---------------------------------------------------------------------


def test_single_successful_call_one_attempt(telemetry, openai_llm):
    openai_llm.client.chat.completions.create.return_value = _openai_response()

    result = openai_llm.chat([{"role": "user", "content": "hi"}])

    assert result == "hello"
    invocations = list(telemetry.llm_invocations.values())
    assert len(invocations) == 1
    invocation = invocations[0]
    assert invocation.status == "success"
    assert invocation.total_attempts == 1
    assert invocation.total_retries == 0
    assert invocation.provider == "openai"
    assert invocation.attempts[0].is_final is True
    assert invocation.attempts[0].provider_request_id == "req_123"


# ---------------------------------------------------------------------
# 2. A failed attempt followed by a successful retry
# ---------------------------------------------------------------------


def test_failed_attempt_then_successful_retry(telemetry, openai_llm):
    openai_llm.client.chat.completions.create.side_effect = [
        TimeoutError("upstream slow"),
        _openai_response(content="recovered"),
    ]

    result = openai_llm.chat([{"role": "user", "content": "hi"}])

    assert result == "recovered"
    invocation = list(telemetry.llm_invocations.values())[0]
    assert invocation.status == "success"
    assert invocation.total_attempts == 2
    assert invocation.total_retries == 1
    assert invocation.total_fallbacks == 0
    assert invocation.attempts[0].status == "error"
    assert invocation.attempts[0].error_type == "timeout"
    assert invocation.attempts[0].is_final is False
    assert invocation.attempts[1].status == "success"
    assert invocation.attempts[1].is_final is True

    # retry_reason belongs to the attempt that was retried INTO, not the
    # one that failed: a failed attempt already explains itself via
    # error_type, so its own retry_reason stays null.
    assert invocation.attempts[0].retry_reason is None
    assert invocation.attempts[1].retry_reason == "timeout"

    # Each attempt has its own stable, unique, ordered ID - retries are never collapsed.
    assert invocation.attempts[0].attempt_id != invocation.attempts[1].attempt_id
    assert invocation.attempts[0].attempt_number == 1
    assert invocation.attempts[1].attempt_number == 2


# ---------------------------------------------------------------------
# 3. Provider / model fallback
# ---------------------------------------------------------------------


def test_provider_or_model_fallback(telemetry):
    """A routing-aware caller (or a future router adapter) can record a
    fallback across attempts using the raw telemetry primitives directly -
    this is the contract an OpenAI-compatible router would drive.

    Sequence: attempt 1 against openai/gpt-4o times out; the router falls
    back to anthropic/claude-3-haiku for attempt 2, which succeeds.
    """
    with llm_invocation(provider="openai", requested_model="gpt-4o", operation="chat") as inv_id:
        try:
            with llm_attempt(provider="openai", requested_model="gpt-4o"):
                raise TimeoutError("primary provider timeout")
        except TimeoutError:
            pass  # router caught it and is about to fall back

        with llm_attempt(
            provider="anthropic",
            requested_model="gpt-4o",
            selected_model="claude-3-haiku-20240307",
            fallback_reason="provider_unavailable",
            routing_reason="primary_timeout",
        ) as fallback_attempt:
            fallback_attempt.input_tokens = 8
            fallback_attempt.output_tokens = 4

    invocation = telemetry.get_llm_invocation(inv_id)
    assert invocation.status == "success"
    assert invocation.total_attempts == 2
    assert invocation.total_fallbacks == 1
    # A fallback is not a retry: attempt 2 switched provider/model rather
    # than retrying the same route, so total_retries stays 0.
    assert invocation.total_retries == 0
    assert invocation.attempts[1].fallback_reason == "provider_unavailable"
    assert invocation.attempts[1].routing_reason == "primary_timeout"
    # retry_reason stays null on a fallback attempt even though the
    # previous attempt failed - fallback_reason already explains it.
    assert invocation.attempts[1].retry_reason is None
    assert invocation.selected_provider == "anthropic"
    assert invocation.selected_model == "claude-3-haiku-20240307"


# ---------------------------------------------------------------------
# 4. Aggregated tokens and cost across all attempts
# ---------------------------------------------------------------------


def test_aggregated_tokens_and_cost_across_attempts(telemetry, openai_llm, monkeypatch):
    from hippocampai import config as config_module

    config_module._config = None
    monkeypatch.setenv("LLM_PRICING", '{"openai:gpt-4o-mini": {"input": 1.0, "output": 2.0}}')

    openai_llm.client.chat.completions.create.side_effect = [
        TimeoutError("slow"),
        _openai_response(prompt_tokens=1_000_000, completion_tokens=500_000),
    ]

    openai_llm.chat([{"role": "user", "content": "hi"}])

    invocation = list(telemetry.llm_invocations.values())[0]
    assert invocation.input_tokens == 1_000_000
    assert invocation.output_tokens == 500_000
    assert invocation.total_tokens == 1_500_000
    # $1/1M input + $2/1M output => $1 + $1 = $2, only from the successful attempt
    assert invocation.estimated_cost == pytest.approx(2.0)

    config_module._config = None


def test_total_only_attempt_usage_is_included_in_invocation(telemetry):
    with llm_invocation(
        provider="router", requested_model="routed-model", operation="chat"
    ) as invocation_id:
        with llm_attempt(provider="router", requested_model="routed-model") as attempt:
            attempt.apply_metadata(UpstreamMetadata(upstream_token_usage={"total_tokens": 37}))

    invocation = telemetry.get_llm_invocation(invocation_id)
    assert invocation.input_tokens == 0
    assert invocation.output_tokens == 0
    assert invocation.total_tokens == 37
    assert telemetry.get_llm_usage_summary()["router"]["total_tokens"] == 37


def test_invocation_token_total_prefers_breakdown_per_attempt(telemetry):
    with llm_invocation(
        provider="router", requested_model="routed-model", operation="chat"
    ) as invocation_id:
        with llm_attempt(provider="router", requested_model="routed-model") as attempt:
            attempt.apply_metadata(
                UpstreamMetadata(
                    upstream_token_usage={
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "total_tokens": 999,
                    }
                )
            )
        with llm_attempt(provider="router", requested_model="routed-model") as attempt:
            attempt.apply_metadata(UpstreamMetadata(upstream_token_usage={"total_tokens": 20}))
        with llm_attempt(provider="router", requested_model="routed-model") as attempt:
            attempt.apply_metadata(
                UpstreamMetadata(upstream_token_usage={"input_tokens": 7, "total_tokens": 500})
            )

    invocation = telemetry.get_llm_invocation(invocation_id)
    assert invocation.input_tokens == 17
    assert invocation.output_tokens == 5
    assert invocation.total_tokens == 42


# ---------------------------------------------------------------------
# 5. User, workspace, tenant, workflow, agent, step attribution
# ---------------------------------------------------------------------


def test_full_attribution_context(telemetry, openai_llm):
    openai_llm.client.chat.completions.create.return_value = _openai_response()

    with llm_trace_context(
        tenant_id="tenant-1",
        workspace_id="workspace-1",
        user_id="user-1",
        session_id="session-1",
        conversation_id="conv-1",
        workflow_id="wf-1",
        workflow_name="Research Pipeline",
        workflow_step_id="step-1",
        workflow_step_name="Draft Outline",
        agent_id="agent-1",
        agent_name="Research Agent",
        tool_name="web_search",
        feature_name="research_agent",
        billing_bucket="team-a",
        metadata={"request_source": "cli"},
    ):
        openai_llm.chat([{"role": "user", "content": "hi"}])

    invocation = list(telemetry.llm_invocations.values())[0]
    assert invocation.tenant_id == "tenant-1"
    assert invocation.workspace_id == "workspace-1"
    assert invocation.user_id == "user-1"
    assert invocation.session_id == "session-1"
    assert invocation.conversation_id == "conv-1"
    assert invocation.workflow_id == "wf-1"
    assert invocation.workflow_name == "Research Pipeline"
    assert invocation.workflow_step_id == "step-1"
    assert invocation.workflow_step_name == "Draft Outline"
    assert invocation.agent_id == "agent-1"
    assert invocation.agent_name == "Research Agent"
    assert invocation.tool_name == "web_search"
    assert invocation.feature_name == "research_agent"
    assert invocation.operation == "research_agent"  # feature_name overrides default op name
    assert invocation.billing_bucket == "team-a"
    assert invocation.metadata.get("request_source") == "cli"

    # Context must not leak to calls made outside the block.
    openai_llm.client.chat.completions.create.return_value = _openai_response()
    openai_llm.chat([{"role": "user", "content": "hi again"}])
    second_invocation = list(telemetry.llm_invocations.values())[1]
    assert second_invocation.workflow_id is None
    assert second_invocation.operation == "chat"


def test_nested_context_merges_without_losing_outer_fields(telemetry, openai_llm):
    openai_llm.client.chat.completions.create.return_value = _openai_response()

    with llm_trace_context(workflow_id="wf-1", agent_id="agent-1"):
        with llm_trace_context(workflow_step_id="step-1"):
            openai_llm.chat([{"role": "user", "content": "hi"}])

    invocation = list(telemetry.llm_invocations.values())[0]
    assert invocation.workflow_id == "wf-1"
    assert invocation.agent_id == "agent-1"
    assert invocation.workflow_step_id == "step-1"


# ---------------------------------------------------------------------
# 6. Missing optional provider metadata
# ---------------------------------------------------------------------


def test_missing_optional_provider_metadata_ollama(telemetry):
    from hippocampai.adapters.provider_ollama import OllamaLLM

    llm = OllamaLLM.__new__(OllamaLLM)
    llm.model = "qwen2.5:7b-instruct"
    llm.base_url = "http://localhost:11434"

    fake_response = MagicMock()
    fake_response.raise_for_status.return_value = None
    # No prompt_eval_count/eval_count/done_reason - some Ollama builds omit these.
    fake_response.json.return_value = {"response": "hi"}

    with patch("httpx.Client") as mock_client_cls:
        mock_client_cls.return_value.__enter__.return_value.post.return_value = fake_response
        result = llm.generate("hello")

    assert result == "hi"
    invocation = list(telemetry.llm_invocations.values())[0]
    assert invocation.status == "success"
    assert invocation.input_tokens == 0
    assert invocation.output_tokens == 0
    assert invocation.attempts[0].provider_request_id is None


def test_missing_optional_metadata_does_not_break_openai(telemetry, openai_llm):
    """A minimal OpenAI-compatible response with no usage object at all."""
    response = MagicMock(spec=["id", "model", "choices"])
    response.id = "req_minimal"
    response.model = "gpt-4o-mini"
    choice = MagicMock()
    choice.finish_reason = "stop"
    choice.message.content = "ok"
    response.choices = [choice]
    openai_llm.client.chat.completions.create.return_value = response

    result = openai_llm.chat([{"role": "user", "content": "hi"}])

    assert result == "ok"
    invocation = list(telemetry.llm_invocations.values())[0]
    assert invocation.status == "success"
    assert invocation.total_tokens == 0


# ---------------------------------------------------------------------
# 7. Unknown model pricing
# ---------------------------------------------------------------------


def test_unknown_model_pricing_returns_none(monkeypatch):
    from hippocampai import config as config_module

    config_module._config = None
    monkeypatch.setenv("LLM_PRICING", "{}")

    cost = estimate_cost(
        provider="openai", model="some-unpriced-model", input_tokens=1000, output_tokens=500
    )
    assert cost is None

    config_module._config = None


def test_cost_estimation_disabled(monkeypatch):
    from hippocampai import config as config_module

    config_module._config = None
    monkeypatch.setenv("LLM_COST_ESTIMATION_ENABLED", "false")
    monkeypatch.setenv("LLM_PRICING", '{"openai:gpt-4o-mini": {"input": 1.0, "output": 2.0}}')

    cost = estimate_cost(
        provider="openai", model="gpt-4o-mini", input_tokens=1_000_000, output_tokens=0
    )
    assert cost is None

    config_module._config = None


# ---------------------------------------------------------------------
# 8. Telemetry disabled
# ---------------------------------------------------------------------


def test_telemetry_disabled_still_returns_result(openai_llm):
    telemetry_module._global_telemetry = None
    disabled_telemetry = get_telemetry(enabled=False)
    try:
        openai_llm.client.chat.completions.create.return_value = _openai_response()
        result = openai_llm.chat([{"role": "user", "content": "hi"}])
        assert result == "hello"
        # No invocations recorded, and it must not raise despite being disabled.
        assert disabled_telemetry.llm_invocations == {}
        assert disabled_telemetry.get_llm_usage_summary() == {}
    finally:
        telemetry_module._global_telemetry = None


# ---------------------------------------------------------------------
# 9. Sanitization of sensitive error data
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,must_not_contain",
    [
        ("Authorization: Bearer sk-abcdefghijklmnop", "sk-abcdefghijklmnop"),
        ("api_key=sk-ant-abcdefgh12345678", "sk-ant-abcdefgh12345678"),
        ("using gsk_abcdefghijklmnopqrstuvwx", "gsk_abcdefghijklmnopqrstuvwx"),
    ],
)
def test_sanitize_error_message_redacts_secrets(raw, must_not_contain):
    sanitized = sanitize_error_message(raw)
    assert must_not_contain not in sanitized
    assert "[REDACTED]" in sanitized


def test_sanitize_error_message_truncates_long_messages():
    sanitized = sanitize_error_message("x" * 5000)
    assert len(sanitized) < 5000
    assert sanitized.endswith("...[truncated]")


def test_end_to_end_error_sanitization_in_invocation(telemetry, openai_llm):
    openai_llm.client.chat.completions.create.side_effect = ConnectionError(
        "failed with Authorization: Bearer sk-live-1234567890abcdef"
    )

    result = openai_llm.chat([{"role": "user", "content": "hi"}])

    assert result == ""
    invocation = list(telemetry.llm_invocations.values())[0]
    assert invocation.status == "error"
    assert "sk-live-1234567890abcdef" not in (invocation.error_message or "")
    for attempt in invocation.attempts:
        assert "sk-live-1234567890abcdef" not in (attempt.error_message or "")


# ---------------------------------------------------------------------
# 10. Synchronous execution
# ---------------------------------------------------------------------


def test_synchronous_calls_do_not_share_context(telemetry, openai_llm):
    openai_llm.client.chat.completions.create.return_value = _openai_response()

    with llm_trace_context(workflow_id="wf-a"):
        openai_llm.chat([{"role": "user", "content": "1"}])

    openai_llm.chat([{"role": "user", "content": "2"}])  # outside any context

    invocations = list(telemetry.llm_invocations.values())
    assert invocations[0].workflow_id == "wf-a"
    assert invocations[1].workflow_id is None


# ---------------------------------------------------------------------
# 11. Asynchronous execution / concurrency isolation
# ---------------------------------------------------------------------


def test_concurrent_threads_do_not_mix_attribution(telemetry, openai_llm_factory=None):
    from hippocampai.adapters.provider_openai import OpenAILLM

    results: dict[str, str] = {}

    def make_llm():
        llm = OpenAILLM.__new__(OpenAILLM)
        llm.model = "gpt-4o-mini"
        llm.client = MagicMock()
        llm.client.chat.completions.create.return_value = _openai_response()
        return llm

    def worker(workflow_id: str):
        llm = make_llm()
        with llm_trace_context(workflow_id=workflow_id):
            llm.chat([{"role": "user", "content": workflow_id}])
        results[workflow_id] = workflow_id

    threads = [threading.Thread(target=worker, args=(f"wf-{i}",)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    invocations = list(telemetry.llm_invocations.values())
    workflow_ids = {inv.workflow_id for inv in invocations}
    assert workflow_ids == {f"wf-{i}" for i in range(5)}
    assert len(invocations) == 5


def test_asyncio_tasks_do_not_mix_attribution(telemetry):
    from hippocampai.adapters.provider_openai import OpenAILLM

    async def worker(workflow_id: str):
        llm = OpenAILLM.__new__(OpenAILLM)
        llm.model = "gpt-4o-mini"
        llm.client = MagicMock()
        llm.client.chat.completions.create.return_value = _openai_response()
        with llm_trace_context(workflow_id=workflow_id):
            await asyncio.sleep(0)  # yield control, exercising real interleaving
            llm.chat([{"role": "user", "content": workflow_id}])

    async def main():
        await asyncio.gather(*(worker(f"wf-{i}") for i in range(5)))

    asyncio.run(main())

    invocations = list(telemetry.llm_invocations.values())
    workflow_ids = {inv.workflow_id for inv in invocations}
    assert workflow_ids == {f"wf-{i}" for i in range(5)}


# ---------------------------------------------------------------------
# 12/13. Streaming completion & cancellation (data-model level)
#
# No provider adapter in this repository implements streaming today, so
# there is no real streaming call path to instrument. The contract still
# carries streaming/time-to-first-token fields for forward compatibility,
# validated directly here.
# ---------------------------------------------------------------------


def test_streaming_flag_and_ttft_are_recorded(telemetry):
    with llm_invocation(
        provider="openai", requested_model="gpt-4o-mini", operation="chat", streaming=True
    ) as invocation_id:
        invocation = telemetry.get_llm_invocation(invocation_id)
        invocation.time_to_first_token_ms = 123.4
        with llm_attempt(provider="openai", requested_model="gpt-4o-mini") as attempt:
            attempt.input_tokens = 10
            attempt.output_tokens = 20

    invocation = telemetry.get_llm_invocation(invocation_id)
    assert invocation.streaming is True
    assert invocation.time_to_first_token_ms == 123.4
    assert invocation.status == "success"


def test_streaming_cancellation_marks_invocation_failed(telemetry):
    with pytest.raises(RuntimeError):
        with llm_invocation(
            provider="openai", requested_model="gpt-4o-mini", operation="chat", streaming=True
        ) as invocation_id:
            with llm_attempt(provider="openai", requested_model="gpt-4o-mini") as attempt:
                attempt.output_tokens = 3  # partial output before cancellation
                raise RuntimeError("stream cancelled by caller")

    invocation = telemetry.get_llm_invocation(invocation_id)
    assert invocation.status == "error"
    assert invocation.attempts[0].status == "error"
    assert invocation.attempts[0].output_tokens == 3  # partial usage preserved


# ---------------------------------------------------------------------
# 14. Backwards compatibility with existing callers
# ---------------------------------------------------------------------


def test_chat_still_returns_plain_string(telemetry, openai_llm):
    openai_llm.client.chat.completions.create.return_value = _openai_response(content="plain")
    result = openai_llm.chat([{"role": "user", "content": "hi"}])
    assert isinstance(result, str)
    assert result == "plain"


def test_generate_delegates_to_chat_as_before(telemetry, openai_llm):
    openai_llm.client.chat.completions.create.return_value = _openai_response(
        content="via generate"
    )
    result = openai_llm.generate("prompt text", system="sys")
    assert result == "via generate"
    # generate() must not create a second, duplicate invocation on top of chat()'s.
    assert len(telemetry.llm_invocations) == 1


def test_openai_swallows_exhausted_errors_as_before(telemetry, openai_llm):
    """OpenAI/Anthropic/Ollama historically return "" rather than raising -
    preserved so existing callers that don't handle exceptions keep working."""
    openai_llm.client.chat.completions.create.side_effect = ConnectionError("down")
    result = openai_llm.chat([{"role": "user", "content": "hi"}])
    assert result == ""


# ---------------------------------------------------------------------
# 15. OpenAI-compatible response metadata extraction
# ---------------------------------------------------------------------


def test_openai_compatible_extraction_with_headers_and_cache_details():
    usage = MagicMock()
    usage.prompt_tokens = 100
    usage.completion_tokens = 50
    usage.total_tokens = 150
    usage.prompt_tokens_details = MagicMock(cached_tokens=40)
    usage.completion_tokens_details = MagicMock(reasoning_tokens=12)

    response = MagicMock()
    response.id = "req_xyz"
    response.model = "gpt-4o"
    response.usage = usage

    headers = {"x-request-id": "hdr-req-1", "openai-processing-ms": "842"}

    meta = UpstreamMetadata.from_openai_response(response, headers=headers)

    assert meta.provider_request_id == "req_xyz"
    assert meta.selected_model == "gpt-4o"
    assert meta.upstream_token_usage["input_tokens"] == 100
    assert meta.upstream_token_usage["output_tokens"] == 50
    assert meta.upstream_token_usage["cached_input_tokens"] == 40
    assert meta.upstream_token_usage["reasoning_tokens"] == 12
    assert meta.provider_metadata["x-request-id"] == "hdr-req-1"
    assert meta.provider_metadata["openai-processing-ms"] == "842"


def test_openai_compatible_extraction_handles_missing_response_gracefully():
    response = MagicMock(spec=["id", "model"])
    response.id = "req_bare"
    response.model = "gpt-4o-mini"
    meta = UpstreamMetadata.from_openai_response(response)
    assert meta.upstream_token_usage == {}
    assert meta.provider_request_id == "req_bare"


# ---------------------------------------------------------------------
# 16. Provider-specific usage normalization
# ---------------------------------------------------------------------


def test_anthropic_usage_normalization():
    usage = MagicMock()
    usage.input_tokens = 200
    usage.output_tokens = 80
    usage.cache_read_input_tokens = 30

    response = MagicMock()
    response.id = "msg_1"
    response.model = "claude-3-5-sonnet-20241022"
    response.usage = usage

    meta = UpstreamMetadata.from_anthropic_response(response)
    assert meta.upstream_token_usage == {
        "input_tokens": 200,
        "output_tokens": 80,
        "cached_input_tokens": 30,
    }
    assert meta.provider_request_id == "msg_1"


def test_ollama_usage_normalization():
    payload = {
        "model": "qwen2.5:7b-instruct",
        "prompt_eval_count": 55,
        "eval_count": 22,
    }
    meta = UpstreamMetadata.from_ollama_response(payload)
    assert meta.upstream_token_usage == {"input_tokens": 55, "output_tokens": 22}
    assert meta.provider_request_id is None  # Ollama is local; no request id
    assert meta.selected_model == "qwen2.5:7b-instruct"


def test_all_providers_normalize_to_same_usage_shape(telemetry):
    """Regardless of provider, an invocation's aggregate usage fields use
    the same names (input_tokens/output_tokens/...) - the whole point of
    the normalized contract."""
    from hippocampai.adapters.provider_anthropic import AnthropicLLM
    from hippocampai.adapters.provider_groq import GroqLLM
    from hippocampai.adapters.provider_openai import OpenAILLM

    # OpenAI
    openai_llm = OpenAILLM.__new__(OpenAILLM)
    openai_llm.model = "gpt-4o-mini"
    openai_llm.client = MagicMock()
    openai_llm.client.chat.completions.create.return_value = _openai_response(
        prompt_tokens=10, completion_tokens=4
    )
    openai_llm.chat([{"role": "user", "content": "hi"}])

    # Groq (OpenAI-compatible)
    groq_llm = GroqLLM.__new__(GroqLLM)
    groq_llm.model = "llama-3.3-70b-versatile"
    groq_llm.client = MagicMock()
    groq_llm.client.chat.completions.create.return_value = _openai_response(
        prompt_tokens=20, completion_tokens=8
    )
    groq_llm.chat([{"role": "user", "content": "hi"}])

    # Anthropic
    anthropic_llm = AnthropicLLM.__new__(AnthropicLLM)
    anthropic_llm.model = "claude-3-5-sonnet-20241022"
    anthropic_llm.client = MagicMock()
    anthropic_usage = MagicMock()
    anthropic_usage.input_tokens = 30
    anthropic_usage.output_tokens = 12
    anthropic_usage.cache_read_input_tokens = None
    anthropic_response = MagicMock()
    anthropic_response.id = "msg_abc"
    anthropic_response.model = "claude-3-5-sonnet-20241022"
    anthropic_response.usage = anthropic_usage
    anthropic_response.stop_reason = "end_turn"
    block = MagicMock()
    block.text = "hi from claude"
    anthropic_response.content = [block]
    anthropic_llm.client.messages.create.return_value = anthropic_response
    anthropic_llm.chat([{"role": "user", "content": "hi"}])

    invocations = list(telemetry.llm_invocations.values())
    assert len(invocations) == 3
    by_provider = {inv.provider: inv for inv in invocations}
    assert by_provider["openai"].input_tokens == 10
    assert by_provider["openai"].output_tokens == 4
    assert by_provider["groq"].input_tokens == 20
    assert by_provider["groq"].output_tokens == 8
    assert by_provider["anthropic"].input_tokens == 30
    assert by_provider["anthropic"].output_tokens == 12
    for inv in invocations:
        assert inv.status == "success"
        assert inv.total_tokens == inv.input_tokens + inv.output_tokens


# ---------------------------------------------------------------------
# Error classification and usage-summary aggregation
# ---------------------------------------------------------------------


def test_classify_llm_error_variants():
    class RateLimitError(Exception):
        pass

    class APITimeoutError(Exception):
        pass

    error_type, retry_reason = classify_llm_error(RateLimitError("429"))
    assert error_type == "rate_limit"
    assert retry_reason == "rate_limit"

    error_type, retry_reason = classify_llm_error(APITimeoutError("timed out"))
    assert error_type == "timeout"
    assert retry_reason == "timeout"

    error_type, retry_reason = classify_llm_error(ConnectionError("refused"))
    assert error_type == "connection_error"
    assert retry_reason == "connection_error"


def test_get_llm_usage_summary_groups_by_dimension(telemetry, openai_llm):
    openai_llm.client.chat.completions.create.return_value = _openai_response(
        prompt_tokens=10, completion_tokens=5
    )
    with llm_trace_context(workflow_id="wf-1"):
        openai_llm.chat([{"role": "user", "content": "1"}])
    with llm_trace_context(workflow_id="wf-1"):
        openai_llm.chat([{"role": "user", "content": "2"}])
    with llm_trace_context(workflow_id="wf-2"):
        openai_llm.chat([{"role": "user", "content": "3"}])

    summary = telemetry.get_llm_usage_summary(group_by="workflow_id")
    assert summary["wf-1"]["invocation_count"] == 2
    assert summary["wf-1"]["total_tokens"] == 30
    assert summary["wf-2"]["invocation_count"] == 1


def test_llm_usage_summary_preserves_all_unknown_costs(telemetry, monkeypatch):
    monkeypatch.setattr(telemetry_module, "estimate_cost", lambda **kwargs: None)

    for _ in range(2):
        with llm_invocation(provider="router", requested_model="model", operation="chat"):
            with llm_attempt(provider="router", requested_model="model"):
                pass

    summary = telemetry.get_llm_usage_summary()["router"]
    assert summary["estimated_cost"] is None
    assert summary["unknown_cost_invocations"] == 2


def test_llm_usage_summary_sums_known_costs_and_counts_unknown(telemetry, monkeypatch):
    costs = iter([1.25, None, 0.0])
    monkeypatch.setattr(telemetry_module, "estimate_cost", lambda **kwargs: next(costs))

    for _ in range(3):
        with llm_invocation(provider="router", requested_model="model", operation="chat"):
            with llm_attempt(provider="router", requested_model="model"):
                pass

    summary = telemetry.get_llm_usage_summary()["router"]
    assert summary["estimated_cost"] == pytest.approx(1.25)
    assert summary["unknown_cost_invocations"] == 1


def test_llm_usage_summary_distinguishes_known_zero_cost(telemetry, monkeypatch):
    monkeypatch.setattr(telemetry_module, "estimate_cost", lambda **kwargs: 0.0)

    with llm_invocation(provider="local", requested_model="model", operation="chat"):
        with llm_attempt(provider="local", requested_model="model"):
            pass

    summary = telemetry.get_llm_usage_summary()["local"]
    assert summary["estimated_cost"] == 0.0
    assert summary["unknown_cost_invocations"] == 0


def test_get_costliest_and_slowest_attempts(telemetry, openai_llm, monkeypatch):
    from hippocampai import config as config_module

    config_module._config = None
    monkeypatch.setenv("LLM_PRICING", '{"openai:gpt-4o-mini": {"input": 5.0, "output": 5.0}}')

    openai_llm.client.chat.completions.create.return_value = _openai_response(
        prompt_tokens=1_000_000, completion_tokens=0
    )
    openai_llm.chat([{"role": "user", "content": "expensive"}])

    openai_llm.client.chat.completions.create.return_value = _openai_response(
        prompt_tokens=1, completion_tokens=0
    )
    openai_llm.chat([{"role": "user", "content": "cheap"}])

    costliest = telemetry.get_costliest_llm_attempts(limit=1)
    assert costliest[0].input_tokens == 1_000_000

    slowest = telemetry.get_slowest_llm_attempts(limit=2)
    assert len(slowest) == 2

    config_module._config = None


def test_llm_trace_context_default_is_empty():
    ctx = LLMTraceContext()
    assert ctx.workflow_id is None
    assert ctx.metadata == {}


def test_metadata_cap_replaces_many_keys_with_bounded_summary(monkeypatch):
    from hippocampai import config as config_module

    config = MagicMock(llm_telemetry_max_metadata_bytes=64)
    monkeypatch.setattr(config_module, "get_config", lambda: config)
    metadata = {f"key-{index}": "value" for index in range(500)}

    capped = _cap_metadata(metadata)

    assert capped == {"_truncated": True, "_key_count": 500}
    assert len(telemetry_module.json.dumps(capped).encode("utf-8")) <= 64


@pytest.mark.parametrize(
    ("max_bytes", "expected"),
    [
        (len('{"_truncated": true}'), {"_truncated": True}),
        (2, {}),
    ],
)
def test_metadata_cap_uses_fallback_that_fits_tiny_budget(monkeypatch, max_bytes, expected):
    from hippocampai import config as config_module

    config = MagicMock(llm_telemetry_max_metadata_bytes=max_bytes)
    monkeypatch.setattr(config_module, "get_config", lambda: config)

    capped = _cap_metadata({"oversized": "x" * 100})

    assert capped == expected
    assert len(telemetry_module.json.dumps(capped).encode("utf-8")) <= max_bytes


def test_metadata_cap_bounds_unserializable_replacement(monkeypatch):
    from hippocampai import config as config_module

    config = MagicMock(llm_telemetry_max_metadata_bytes=2)
    monkeypatch.setattr(config_module, "get_config", lambda: config)
    circular = {}
    circular["self"] = circular

    capped = _cap_metadata(circular)

    assert capped == {}
    assert len(telemetry_module.json.dumps(capped).encode("utf-8")) <= 2


def test_memory_telemetry_llm_invocations_isolated_per_instance():
    a = MemoryTelemetry(enabled=True)
    b = MemoryTelemetry(enabled=True)
    a.start_llm_invocation(provider="openai", requested_model="gpt-4o-mini", operation="chat")
    assert len(a.llm_invocations) == 1
    assert len(b.llm_invocations) == 0


# ---------------------------------------------------------------------
# Request identity: HippocampAI request_id vs. router_request_id vs.
# provider_request_id must never be conflated.
# ---------------------------------------------------------------------


def test_hippocampai_request_id_is_authoritative_and_unaffected_by_router_metadata(
    telemetry, openai_llm
):
    """HippocampAI's own request_id must never be overwritten by anything
    a router or provider supplies."""
    response = _openai_response()
    response.id = "chatcmpl-provider-side-id"
    openai_llm.client.chat.completions.create.return_value = response

    openai_llm.chat([{"role": "user", "content": "hi"}])

    invocation = list(telemetry.llm_invocations.values())[0]
    assert invocation.request_id
    # HippocampAI's own logical request_id is a UUID it minted itself, not
    # the provider's response id or anything router-supplied.
    assert invocation.request_id != "chatcmpl-provider-side-id"
    assert invocation.attempts[0].provider_request_id == "chatcmpl-provider-side-id"


def test_router_request_id_is_preserved_separately_from_hippocampai_request_id(telemetry):
    """A router-supplied identity is recorded on the attempt (and rolled up
    onto the invocation) without ever touching HippocampAI's own request_id
    or the attempt's provider_request_id."""
    with llm_invocation(
        provider="openai", requested_model="gpt-4o-mini", operation="chat"
    ) as inv_id:
        with llm_attempt(provider="openai", requested_model="gpt-4o-mini") as attempt:
            attempt.apply_metadata(
                UpstreamMetadata(
                    router_request_id="router-abc-123",
                    provider_request_id="chatcmpl-real-provider-id",
                )
            )

    invocation = telemetry.get_llm_invocation(inv_id)
    assert invocation.router_request_id == "router-abc-123"
    assert invocation.attempts[0].router_request_id == "router-abc-123"
    assert invocation.attempts[0].provider_request_id == "chatcmpl-real-provider-id"
    # The three identities are distinct values, not aliases of each other.
    assert invocation.request_id != invocation.router_request_id
    assert invocation.router_request_id != invocation.attempts[0].provider_request_id


def test_router_request_id_never_copied_into_provider_request_id(telemetry):
    with llm_invocation(provider="openai", requested_model="gpt-4o-mini", operation="chat"):
        with llm_attempt(provider="openai", requested_model="gpt-4o-mini") as attempt:
            # A router supplies only its own gateway-level ID - no
            # per-attempt provider ID is available from it.
            attempt.apply_metadata(UpstreamMetadata(router_request_id="router-only-id"))

    invocation = list(telemetry.llm_invocations.values())[0]
    assert invocation.attempts[0].router_request_id == "router-only-id"
    assert invocation.attempts[0].provider_request_id is None


def test_upstream_metadata_request_id_is_deprecated_alias_for_router_request_id(telemetry):
    """Backwards compatibility: callers still using the old
    UpstreamMetadata.request_id field get the same router_request_id
    behavior, never mapped into provider_request_id."""
    with llm_invocation(provider="openai", requested_model="gpt-4o-mini", operation="chat"):
        with llm_attempt(provider="openai", requested_model="gpt-4o-mini") as attempt:
            attempt.apply_metadata(UpstreamMetadata(request_id="legacy-router-id"))

    invocation = list(telemetry.llm_invocations.values())[0]
    assert invocation.attempts[0].router_request_id == "legacy-router-id"
    assert invocation.attempts[0].provider_request_id is None

    # router_request_id (new field) takes precedence when both are set.
    with llm_invocation(provider="openai", requested_model="gpt-4o-mini", operation="chat"):
        with llm_attempt(provider="openai", requested_model="gpt-4o-mini") as attempt:
            attempt.apply_metadata(
                UpstreamMetadata(request_id="legacy-id", router_request_id="preferred-id")
            )
    invocation2 = list(telemetry.llm_invocations.values())[1]
    assert invocation2.attempts[0].router_request_id == "preferred-id"


def test_provider_request_id_stays_none_when_not_supplied(telemetry):
    """Missing provider_request_id is valid and must not break tracing or
    aggregation - it is never invented."""
    with llm_invocation(provider="ollama", requested_model="qwen2.5:7b-instruct", operation="chat"):
        with llm_attempt(provider="ollama", requested_model="qwen2.5:7b-instruct"):
            pass  # local Ollama call: no provider request id, no router id

    invocation = list(telemetry.llm_invocations.values())[0]
    assert invocation.status == "success"
    assert invocation.attempts[0].provider_request_id is None
    assert invocation.attempts[0].router_request_id is None
    assert invocation.router_request_id is None


def test_serialized_trace_preserves_router_request_id(telemetry):
    with llm_invocation(
        provider="openai", requested_model="gpt-4o-mini", operation="chat"
    ) as inv_id:
        with llm_attempt(provider="openai", requested_model="gpt-4o-mini") as attempt:
            attempt.apply_metadata(UpstreamMetadata(router_request_id="router-export-check"))

    exported = telemetry.export_llm_invocations([inv_id])[0]
    assert exported["router_request_id"] == "router-export-check"
    assert exported["attempts"][0]["router_request_id"] == "router-export-check"
    assert exported["request_id"] != exported["router_request_id"]


# ---------------------------------------------------------------------
# Retry vs. fallback accounting must not double-count.
# ---------------------------------------------------------------------


def test_pure_retry_increments_total_retries_only(telemetry):
    with llm_invocation(
        provider="openai", requested_model="gpt-4o-mini", operation="chat"
    ) as inv_id:
        try:
            with llm_attempt(provider="openai", requested_model="gpt-4o-mini"):
                raise TimeoutError("slow")
        except TimeoutError:
            pass
        with llm_attempt(provider="openai", requested_model="gpt-4o-mini"):
            pass  # same route, succeeds

    invocation = telemetry.get_llm_invocation(inv_id)
    assert invocation.total_attempts == 2
    assert invocation.total_retries == 1
    assert invocation.total_fallbacks == 0


def test_pure_fallback_increments_total_fallbacks_only(telemetry):
    with llm_invocation(
        provider="groq", requested_model="llama-3.3-70b-versatile", operation="chat"
    ) as inv_id:
        try:
            with llm_attempt(provider="groq", requested_model="llama-3.3-70b-versatile"):
                raise TimeoutError("rate limited")
        except TimeoutError:
            pass
        with llm_attempt(
            provider="openai",
            requested_model="llama-3.3-70b-versatile",
            selected_model="gpt-4o-mini",
            fallback_reason="provider_rate_limited",
        ):
            pass

    invocation = telemetry.get_llm_invocation(inv_id)
    assert invocation.total_attempts == 2
    assert invocation.total_retries == 0
    assert invocation.total_fallbacks == 1


def test_retry_then_fallback_counts_each_exactly_once(telemetry):
    """Scenario D: attempt 1 fails, attempt 2 retries the same route and
    also fails, attempt 3 falls back and succeeds. total_retries=1,
    total_fallbacks=1, total_attempts=3 - never total_retries=2."""
    with llm_invocation(
        provider="groq", requested_model="llama-3.3-70b-versatile", operation="chat"
    ) as inv_id:
        try:
            with llm_attempt(provider="groq", requested_model="llama-3.3-70b-versatile"):
                raise TimeoutError("timeout 1")
        except TimeoutError:
            pass
        try:
            with llm_attempt(provider="groq", requested_model="llama-3.3-70b-versatile"):
                raise TimeoutError("timeout 2")
        except TimeoutError:
            pass
        with llm_attempt(
            provider="openai",
            requested_model="llama-3.3-70b-versatile",
            selected_model="gpt-4o-mini",
            fallback_reason="provider_rate_limited",
        ):
            pass

    invocation = telemetry.get_llm_invocation(inv_id)
    assert invocation.total_attempts == 3
    assert invocation.total_retries == 1
    assert invocation.total_fallbacks == 1
    assert invocation.status == "success"


# ---------------------------------------------------------------------
# Partial upstream telemetry: missing fields stay None, never fabricated
# as zero.
# ---------------------------------------------------------------------


def test_missing_external_telemetry_remains_none_not_zero(telemetry):
    """A router that only exposes a router_request_id (no per-attempt
    latency/tokens/provider id) must not have those fields silently
    replaced with 0 - that would look like a measured zero."""
    with llm_invocation(provider="openai", requested_model="gpt-4o-mini", operation="chat"):
        with llm_attempt(provider="openai", requested_model="gpt-4o-mini") as attempt:
            attempt.apply_metadata(UpstreamMetadata(router_request_id="router-partial"))
            # No input_tokens/output_tokens/provider_request_id supplied.

    invocation = list(telemetry.llm_invocations.values())[0]
    attempt = invocation.attempts[0]
    assert attempt.input_tokens is None
    assert attempt.output_tokens is None
    assert attempt.provider_request_id is None
    # The invocation-level aggregate correctly treats the missing per-attempt
    # usage as contributing nothing, without claiming the attempt itself
    # measured zero tokens.
    assert invocation.input_tokens == 0
    assert invocation.output_tokens == 0


# ---------------------------------------------------------------------
# Example fixture: examples/traces/provider_trace_example.json must stay
# schema-valid against the live implementation.
# ---------------------------------------------------------------------


def test_example_fixture_matches_live_schema():
    import json
    from dataclasses import fields
    from pathlib import Path

    from hippocampai.telemetry import LLMInvocation, UpstreamAttempt

    fixture_path = (
        Path(__file__).resolve().parents[1] / "examples" / "traces" / "provider_trace_example.json"
    )
    fixture = json.loads(fixture_path.read_text())

    invocation_fields = {f.name for f in fields(LLMInvocation)} - {"attempts"}
    attempt_fields = {f.name for f in fields(UpstreamAttempt)}

    assert set(fixture.keys()) - {"attempts"} == invocation_fields
    assert len(fixture["attempts"]) == 2
    for attempt in fixture["attempts"]:
        assert set(attempt.keys()) == attempt_fields

    # Sanitization: no secret-shaped content anywhere in the fixture.
    raw = json.dumps(fixture)
    assert "sk-" not in raw
    assert "Bearer " not in raw or "[REDACTED]" in raw
    assert "router_request_id" in fixture
    assert fixture["attempts"][0]["retry_reason"] is None or isinstance(
        fixture["attempts"][0]["retry_reason"], str
    )
