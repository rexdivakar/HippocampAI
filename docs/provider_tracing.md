# Provider-Neutral LLM Trace Contract: Schema Reference

This document is the schema reference for HippocampAI's LLM usage tracing,
prepared for external review (see
[GitHub Discussion #11](https://github.com/rexdivakar/HippocampAI/discussions/11)).
It documents exactly what is implemented today. Every field named below
exists in `src/hippocampai/telemetry.py` as of this writing; nothing here is
aspirational.

**Naming note:** this document uses the conceptual names discussed in #11
(`TraceContext`, `LogicalInvocation`, `ProviderAttempt`) as section headings,
since that's the vocabulary the design discussion uses. The actual Python
classes are named `LLMTraceContext`, `LLMInvocation`, and `UpstreamAttempt`
(same fields, same semantics; see [Schema Reference](#schema-reference) for
the exact file/class mapping).

For usage examples and how-to guidance, see [`TELEMETRY.md`](TELEMETRY.md#llm-usage-tracing).
For the runnable validation scenario referenced below, see
[`scripts/validate_llm_tracing.py`](../scripts/validate_llm_tracing.py).

---

## Overview

HippocampAI is the source of truth for:

- **Workflow identity** (`workflow_id`, `workflow_name`)
- **Workflow Step identity** (`workflow_step_id`, `workflow_step_name`)
- **Agent identity** (`agent_id`, `agent_name`)
- **Tool identity** (`tool_name`)
- **Feature identity** (`feature_name`)

These are always assigned by the application code calling HippocampAI (via
`llm_trace_context`, see below), never by a provider or a routing layer.

An external OpenAI-compatible routing layer (NovaRouteAI or any other) **may
optionally** supply, per upstream attempt:

| Field | Populated via |
|---|---|
| `logical_request_id`* | n/a; always assigned by HippocampAI (see [Note on `logical_request_id`](#note-on-logical_request_id)) |
| `provider_request_id` | `UpstreamMetadata.provider_request_id` |
| `attempt_id`* | n/a; always assigned by HippocampAI |
| `selected_provider`* | the attempt's own `provider` field, set by the caller |
| `selected_model` | `UpstreamMetadata.selected_model` |
| `routing_reason` | `UpstreamMetadata.routing_reason` |
| `retry_reason` | `UpstreamMetadata.retry_reason` |
| `fallback_reason` | `UpstreamMetadata.fallback_reason` |
| `latency_ms` | `UpstreamMetadata.upstream_latency_ms` (advisory; HippocampAI also measures its own wall-clock latency around the call regardless) |
| token usage (`input`/`output`/`cached_input`/`reasoning`) | `UpstreamMetadata.upstream_token_usage` |
| provider metadata | `UpstreamMetadata.provider_metadata` |

\* `logical_request_id` and `attempt_id` are always minted by HippocampAI, not
the router: they identify *HippocampAI's own record*, not the upstream
request. `selected_provider` is likewise the caller's own declaration of
which provider actually served the attempt (there's no dedicated field for a
router to *report* the provider back; the caller states it when opening the
attempt). This is intentional: identity fields are never delegated to an
external system.

**Every field a router can supply is optional.** `UpstreamMetadata` fields
default to `None`/empty, and `UpstreamAttempt.apply_metadata()` only
overwrites a field when the router actually supplied it. A provider or
router that exposes none of this continues to function identically to one
that exposes all of it. No provider integration in this repository, and no
routing layer, is required for tracing to work; the four built-in adapters
(OpenAI, Anthropic, Groq, Ollama) populate `UpstreamMetadata` themselves from
each SDK's native response shape.

---

## TraceContext

**Class:** `LLMTraceContext` (`src/hippocampai/telemetry.py`), an immutable
(`frozen=True`) dataclass, propagated via `contextvars.ContextVar` so it is
safe across synchronous code, threads, and asyncio tasks without any global
mutable state. Attached to a block of code with the `llm_trace_context(...)`
context manager; every LLM call made inside that block inherits it
automatically.

| Field | Type | Required | Owner | Description |
|---|---|---|---|---|
| `trace_id` | `Optional[str]` | Optional | HippocampAI (caller-suppliable) | Correlates multiple logical invocations under one caller-defined trace. Auto-generated per invocation if not supplied. |
| `parent_span_id` | `Optional[str]` | Optional | Caller | Pass-through link to an external span/trace system, if any. Not interpreted by HippocampAI. |
| `tenant_id` | `Optional[str]` | Optional | Caller | Multi-tenant attribution. |
| `workspace_id` | `Optional[str]` | Optional | Caller | Workspace-level attribution. |
| `user_id` | `Optional[str]` | Optional | Caller | End-user attribution. |
| `session_id` | `Optional[str]` | Optional | Caller | Session-level attribution. |
| `conversation_id` | `Optional[str]` | Optional | Caller | Conversation-level attribution. |
| `workflow_id` | `Optional[str]` | Optional | **HippocampAI** | Stable workflow identifier. |
| `workflow_name` | `Optional[str]` | Optional | **HippocampAI** | Human-readable workflow label. |
| `workflow_step_id` | `Optional[str]` | Optional | **HippocampAI** | Stable workflow-step identifier. |
| `workflow_step_name` | `Optional[str]` | Optional | **HippocampAI** | Human-readable step label. |
| `agent_id` | `Optional[str]` | Optional | **HippocampAI** | Stable agent identifier. |
| `agent_name` | `Optional[str]` | Optional | **HippocampAI** | Human-readable agent label. |
| `tool_name` | `Optional[str]` | Optional | **HippocampAI** | Set when the call originated from a tool invocation. |
| `feature_name` | `Optional[str]` | Optional | **HippocampAI** | Logical operation/prompt name; also becomes the invocation's `operation` field when set. |
| `billing_bucket` | `Optional[str]` | Optional | Caller | Arbitrary cost-attribution bucket (e.g. team, project, plan tier). |
| `metadata` | `dict[str, Any]` | Optional | Caller | Free-form custom metadata, size-capped by `LLM_TELEMETRY_MAX_METADATA_BYTES` (default 8192 bytes; oversized metadata is replaced with a truncation marker, never silently dropped without a trace). |

**Every field is optional.** Nothing in the trace pipeline requires any of
them to be set. A call made with no `llm_trace_context` at all is still
fully traced, just with `None` in every attribution field.

Nested `llm_trace_context(...)` blocks **merge**, not replace: an inner
block's explicit fields override the outer block's, but unset fields (and
`metadata` keys) are inherited from the outer block. This is how a workflow
sets `workflow_id`/`agent_id` once and individual steps only need to add
`workflow_step_id`.

---

## LogicalInvocation

**Class:** `LLMInvocation` (`src/hippocampai/telemetry.py`), representing
**one** LLM operation as requested by application code (one
`client.llm.chat(...)` or `.generate(...)` call). Created and finalized
automatically by the `llm_invocation(...)` context manager wrapping each
provider adapter's public method.

| Field | Type | Required | Description |
|---|---|---|---|
| `invocation_id` | `str` | Always set | Internal record identifier. |
| `trace_id` | `str` | Always set | From `TraceContext.trace_id`, or auto-generated. |
| `request_id` | `str` | Always set | The **logical request ID** (see [note](#note-on-logical_request_id)): a fresh UUID minted per invocation, never shared across invocations or reused from a provider response. |
| `parent_span_id` | `Optional[str]` | Optional | From `TraceContext`. |
| `provider` | `str` | Always set | The provider **requested** (e.g. `"openai"`), from the first attempt. |
| `requested_model` | `str` | Always set | The model **requested**, independent of what a fallback attempt ends up using. |
| `operation` | `str` | Always set | The traced operation name: `TraceContext.feature_name` if set, else the adapter method name (`"chat"`/`"generate"`). |
| `workflow_id`, `workflow_name`, `workflow_step_id`, `workflow_step_name`, `agent_id`, `agent_name`, `tool_name`, `feature_name`, `tenant_id`, `workspace_id`, `user_id`, `session_id`, `conversation_id`, `billing_bucket` | `Optional[str]` | Optional | Copied from the active `TraceContext` at invocation start. |
| `streaming` | `bool` | Always set (default `False`) | Whether this invocation was a streaming call. No adapter in this repository streams today (see [Streaming](#streaming-status)); the field exists for forward compatibility. |
| `time_to_first_token_ms` | `Optional[float]` | Optional | Set by a streaming caller on first-chunk receipt. Unused by current adapters. |
| `start_time` / `end_time` | `datetime` / `Optional[datetime]` | `start_time` always set | Wall-clock bounds of the whole logical call. |
| `latency_ms` | `Optional[float]` | Set on finalize | `end_time - start_time`, in milliseconds. Spans **every** attempt, not just the winning one. |
| `selected_provider` / `selected_model` | `Optional[str]` | Set on finalize | The provider/model of the **final** (`is_final=True`) attempt. |
| `input_tokens` / `output_tokens` / `cached_input_tokens` / `reasoning_tokens` / `total_tokens` | `int` | Default `0`, set on finalize | Sum across **all** attempts that reported usage (including failed ones that returned partial usage before erroring). |
| `estimated_cost` | `Optional[float]` | Set on finalize | Sum of every attempt's own `estimated_cost`; `None` if no attempt had a known cost. Never a fabricated number (see [Aggregation Rules](#aggregation-rules)). |
| `status` | `str` | Default `"in_progress"` | `"in_progress"` \| `"success"` \| `"error"`. See [Status values](#status-values). |
| `error_type` / `error_message` | `Optional[str]` | Set on error | Sanitized error classification/message for the invocation as a whole. |
| `finish_reason` | `Optional[str]` | Optional | Inherited from the final attempt's `finish_reason` unless explicitly overridden. |
| `total_attempts` | `int` | Set on finalize | `len(attempts)`. |
| `total_retries` | `int` | Set on finalize | `max(0, total_attempts - 1)`. |
| `total_fallbacks` | `int` | Set on finalize | Count of attempts whose `fallback_reason` is set. |
| `attempts` | `list[ProviderAttempt]` | Always present (may be empty) | **Every** upstream attempt, in order. See [ProviderAttempt](#providerattempt). |
| `metadata` | `dict[str, Any]` | Optional | From `TraceContext.metadata`, size-capped as above. |

### Relationship to `ProviderAttempt`

`LLMInvocation.attempts` is an ordered list of `UpstreamAttempt` records.
`finalize()` (called automatically when the traced call returns or raises)
computes every aggregate field above from that list. It never estimates or
approximates; if the list is empty (only possible if telemetry recording
itself failed, which is swallowed and logged rather than raised), aggregates
default to `0`/`None`.

### Note on `logical_request_id`

The design discussion in #11 names this `logical_request_id`. In the
implementation it is the `LLMInvocation.request_id` field. It is distinct
from `ProviderAttempt.provider_request_id`, which is whatever ID (if any)
the actual upstream provider returned for that specific attempt. The two
are never conflated.

---

## ProviderAttempt

**Class:** `UpstreamAttempt` (`src/hippocampai/telemetry.py`), representing
**one real request** made to an LLM provider. A single `LogicalInvocation`
may contain several of these; **retries are never collapsed**. Each retry,
rate-limit backoff, timeout, or model/provider fallback produces its own
`ProviderAttempt` record, appended to `LogicalInvocation.attempts` in order.

| Field | Type | Required | Description |
|---|---|---|---|
| `attempt_id` | `str` | Always set | Unique ID for this specific attempt (fresh UUID). Stable and ordered: `attempt_number` gives the 1-based order within the invocation. |
| `attempt_number` | `int` | Always set | `1` for the first attempt, `2` for the first retry, etc. |
| `provider` | `str` | Always set | The provider actually contacted for **this** attempt (may differ from the invocation's `provider` after a provider fallback). |
| `requested_model` | `str` | Always set | The model that was asked for on this attempt. |
| `selected_model` | `Optional[str]` | Optional | The model that actually served the request, if a router/provider reports it (may differ from `requested_model` on a model fallback). |
| `provider_request_id` | `Optional[str]` | Optional | The upstream provider's own request/response ID, if returned (e.g. OpenAI's `response.id`). `None` for providers that don't expose one (e.g. local Ollama). |
| `start_time` / `end_time` | `datetime` / `Optional[datetime]` | `start_time` always set | Wall-clock bounds of this specific attempt only. |
| `latency_ms` | `Optional[float]` | Set on completion | This attempt's own latency, never the whole invocation's. |
| `input_tokens` / `output_tokens` / `cached_input_tokens` / `reasoning_tokens` / `total_tokens` | `Optional[int]` | Optional | Usage for this attempt only, normalized from the provider's native response shape. `None` (not `0`) when the provider didn't report it. |
| `estimated_cost` | `Optional[float]` | Optional | This attempt's own estimated cost; `None` if pricing is unknown for `(provider, selected_model or requested_model)`. |
| `status` | `str` | Default `"in_progress"` | `"in_progress"` \| `"success"` \| `"error"`. See [Status values](#status-values). |
| `finish_reason` | `Optional[str]` | Optional | Provider-native finish/stop reason (e.g. `"stop"`, `"length"`), when returned. |
| `retry_reason` | `Optional[str]` | Optional | Why this attempt is a retry of a prior one (e.g. `"rate_limit"`, `"timeout"`, `"connection_error"`, `"server_error"`). Auto-classified from the exception when not explicitly supplied. |
| `fallback_reason` | `Optional[str]` | Optional | Set only when this attempt represents a provider/model fallback rather than a same-provider retry. Presence of this field is what `total_fallbacks` counts. |
| `routing_reason` | `Optional[str]` | Optional | Free-form explanation from a router for why this attempt was routed the way it was. |
| `is_final` | `bool` | Default `False` | `True` only on the attempt that actually completed the logical invocation (successfully, or as the last attempt before giving up). |
| `error_type` | `Optional[str]` | Set on error | Sanitized error classification (see [Privacy](#privacy)), e.g. `"rate_limit"`, `"timeout"`, `"connection_error"`, `"auth_error"`, `"not_found"`, `"server_error"`, `"api_error"`, `"invalid_request"`, or an exception-derived fallback. Never the raw exception type/message. |
| `error_message` | `Optional[str]` | Set on error | Sanitized error text (`sanitized_error` in reviewer terminology). See [Privacy](#privacy). |
| `metadata` | `dict[str, Any]` | Optional | Provider-specific metadata (the reviewer's `provider_metadata`), e.g. rate-limit headers, populated via `UpstreamMetadata.provider_metadata`. |

### Status values

Both `LLMInvocation.status` and `UpstreamAttempt.status` use the same
three-value set: `"in_progress"` (default, before completion),
`"success"`, `"error"`. There is no separate `terminal_outcome` field. An
attempt's **terminal outcome** is the combination of `status` and
`is_final`: `is_final=True` marks the attempt that ended the invocation
(whether by succeeding, or by being the last attempt before the invocation
gave up).

### Retry behavior

A retry (rate limit, timeout, transient connection/server error) produces a
new `ProviderAttempt` with an incremented `attempt_number`, the same
`provider`/`requested_model` as the one it's retrying, and a `retry_reason`
either supplied explicitly or auto-classified from the causing exception.
`LogicalInvocation.total_retries` is always `total_attempts - 1`.

### Fallback behavior

A fallback (provider or model) is any `ProviderAttempt` with `fallback_reason`
set. There is no structural difference from a retry beyond that field:
`provider`/`requested_model`/`selected_model` may simply differ from the
previous attempt's. `LogicalInvocation.total_fallbacks` counts attempts with
`fallback_reason` set.

---

## Relationship Diagram

```
Workflow (workflow_id, workflow_name)
  │
  ▼
Workflow Step (workflow_step_id, workflow_step_name)
  │
  ▼
Logical Invocation  (request_id = "logical request id", trace_id, provider,
                      requested_model, attribution copied from TraceContext)
  │
  ├─▶ Attempt 1  (attempt_id, attempt_number=1, status="error",
  │               retry_reason="rate_limit", is_final=false)
  │
  ├─▶ Attempt 2  (attempt_id, attempt_number=2, status="error",
  │               retry_reason="timeout", is_final=false)
  │
  └─▶ Attempt N  (attempt_id, attempt_number=N, status="success",
                  fallback_reason="provider_rate_limited", is_final=true)

Logical Invocation aggregates:
  total_attempts = N
  total_retries  = N - 1
  total_fallbacks = attempts with fallback_reason set
  input_tokens / output_tokens / total_tokens = sum across all attempts
  estimated_cost = sum of attempt costs (None if none priced)
  latency_ms     = end_time - start_time (spans every attempt)
  selected_provider / selected_model = from the final (is_final=true) attempt
  status         = "success" | "error"
```

- **One** `Workflow` → **many** `Workflow Step`s (identified by `workflow_id`/`workflow_step_id` on the shared `TraceContext`; HippocampAI does not maintain a separate workflow/step registry. They are caller-supplied identifiers threaded through every invocation and attempt within that context).
- **One** `Workflow Step` → **many** `Logical Invocation`s (every LLM call made while that step's context is active).
- **One** `Logical Invocation` → **one or more** `Provider Attempt`s, always at least one, ordered by `attempt_number`.

---

## Aggregation Rules

All aggregation happens once, in `LLMInvocation.finalize()`, from the
`attempts` list, never estimated, always computed directly from what was
recorded:

- **Latency**: `latency_ms = (end_time - start_time)` measured around the
  *entire* logical invocation (from before attempt 1 starts to after the
  final attempt completes). This is why it is always `>=` the sum of
  individual attempt latencies, not merely equal to the winning attempt's
  latency.
- **Token usage**: `input_tokens`, `output_tokens`, `cached_input_tokens`,
  `reasoning_tokens` are each the arithmetic sum of that field across every
  attempt that reported it (an attempt with `None` for a given field
  contributes `0`). `total_tokens = input_tokens + output_tokens`.
- **Retries**: `total_retries = total_attempts - 1` (a single successful
  attempt has zero retries).
- **Fallbacks**: `total_fallbacks = count(attempts where fallback_reason is set)`.
- **Estimated cost**: `estimated_cost = sum(attempt.estimated_cost for attempt
  in attempts if attempt.estimated_cost is not None)`, or `None` if **no**
  attempt had a known price (see [Cost Estimation](TELEMETRY.md#cost-estimation)).
  Pricing is opt-in configuration, never a built-in catalog, so "unknown"
  is a real, honest outcome, not an error.
- **Final provider/model**: taken from the attempt with `is_final=True` (the
  last attempt appended, whether it succeeded or was the last try before
  giving up).

---

## Privacy

**Stored by default (operational metadata only):**
identifiers (trace/request/attempt IDs, workflow/step/agent/tool/tenant/
workspace/user/session/conversation IDs), provider/model names, token
counts, latency, status, retry/fallback/routing reasons, finish reason,
sanitized error type/message, and caller-supplied custom metadata
(size-capped).

**Never stored, by default and with no opt-in flag to change it:**
- Prompt content (the text sent to the model)
- Model output / completion content
- API keys
- `Authorization`/`Bearer` header values
- Any other secret

**Sanitization rules** (`sanitize_error_message()` / `classify_llm_error()`
in `telemetry.py`), applied to every error before it is written to a trace:
1. Pattern-based redaction of common secret shapes: `sk-...`, `sk-ant-...`,
   `gsk_...` API key prefixes, `Bearer <token>`, and `api_key=`/
   `authorization=` style key-value pairs, each replaced with `[REDACTED]`.
2. Truncation to 500 characters (providers occasionally echo request
   context into error bodies; truncation bounds the blast radius even if a
   novel secret pattern isn't matched by rule 1).
3. Error **type** classification is done by inspecting the Python exception
   class name/module (e.g. `RateLimitError` → `"rate_limit"`), never by
   including the raw exception's `str()` in the type field.

This is not an opt-in feature. It runs unconditionally on every recorded
error, in every provider adapter.

---

## OpenAI Compatible Contract

**Class:** `UpstreamMetadata` (`src/hippocampai/telemetry.py`). A small,
provider-neutral dataclass that any OpenAI-compatible upstream (a hosted
provider, a self-hosted proxy, or a routing layer) can populate to enrich a
`ProviderAttempt`. HippocampAI has no dependency on, and no special-cased
logic for, any specific routing product; this contract is generic.

| Field | Type | Maps to `ProviderAttempt` field |
|---|---|---|
| `request_id` | `Optional[str]` | (advisory; not currently surfaced separately from `provider_request_id`) |
| `provider_request_id` | `Optional[str]` | `provider_request_id` |
| `selected_model` | `Optional[str]` | `selected_model` |
| `routing_reason` | `Optional[str]` | `routing_reason` |
| `retry_reason` | `Optional[str]` | `retry_reason` |
| `fallback_reason` | `Optional[str]` | `fallback_reason` |
| `upstream_attempt` | `Optional[int]` | (advisory; HippocampAI's own `attempt_number` is authoritative) |
| `upstream_latency_ms` | `Optional[float]` | (advisory; HippocampAI measures its own wall-clock latency independently) |
| `upstream_token_usage` | `dict[str, int]` (`input_tokens`/`output_tokens`/`cached_input_tokens`/`reasoning_tokens`/`total_tokens`) | `input_tokens`, `output_tokens`, `cached_input_tokens`, `reasoning_tokens`, `total_tokens` |
| `provider_metadata` | `dict[str, Any]` | merged into `metadata` |

Built-in normalizers already implement this contract for the four supported
providers:

- `UpstreamMetadata.from_openai_response(response, headers=None)`: OpenAI
  and any OpenAI-compatible client (Groq uses this same normalizer).
- `UpstreamMetadata.from_anthropic_response(response)`: Anthropic Messages API.
- `UpstreamMetadata.from_ollama_response(payload)`: Ollama's local JSON
  payload (no request ID; local inference has none).

A future routing layer adopts the same contract: construct an
`UpstreamMetadata` (or call one of the `from_*_response` classmethods
against its own OpenAI-compatible response) and pass it to
`attempt.apply_metadata(...)` inside an `llm_attempt(...)` block. Every field
is optional. See [`UpstreamAttempt.apply_metadata()`](../src/hippocampai/telemetry.py),
which only overwrites fields actually present in the supplied metadata.

---

## Streaming status

No provider adapter in this repository (`OpenAILLM`, `AnthropicLLM`,
`GroqLLM`, `OllamaLLM`) implements streaming today: `chat()`/`generate()`
return a plain `str`. `LLMInvocation.streaming` and
`time_to_first_token_ms` exist in the schema for forward compatibility and
are validated at the data-model level (see `tests/test_llm_telemetry.py`),
but no cancelled/failed-stream behavior exists to document beyond the
general error path, since there is no streaming code path yet.

---

## Schema Reference

For external reviewers navigating the source directly:

| Conceptual name (Discussion #11) | Class | File | Purpose |
|---|---|---|---|
| `TraceContext` | `LLMTraceContext` | `src/hippocampai/telemetry.py` | Immutable, `contextvars`-propagated attribution context (workflow/step/agent/tool/tenant/... identity) attached to a block of code via `llm_trace_context(...)`. |
| `LogicalInvocation` | `LLMInvocation` | `src/hippocampai/telemetry.py` | One LLM operation as requested by application code; aggregates every `ProviderAttempt` it took. |
| `ProviderAttempt` | `UpstreamAttempt` | `src/hippocampai/telemetry.py` | One real request to an LLM provider; one or more per `LogicalInvocation`. |
| OpenAI-compatible contract | `UpstreamMetadata` | `src/hippocampai/telemetry.py` | Optional, provider-neutral metadata payload any OpenAI-compatible upstream can supply to enrich a `ProviderAttempt`. |
| (n/a; collector) | `MemoryTelemetry` | `src/hippocampai/telemetry.py` | The existing telemetry singleton (`get_telemetry()`), extended with `llm_invocations: dict[str, LLMInvocation]` and query/aggregation methods (`get_llm_usage_summary`, `get_recent_llm_invocations`, `get_costliest_llm_attempts`, `get_slowest_llm_attempts`, `export_llm_invocations`). |

Related, not part of the trace schema itself:

| Component | File | Purpose |
|---|---|---|
| `llm_trace_context(...)` | `src/hippocampai/telemetry.py` | Context manager to attach a `TraceContext`. |
| `llm_invocation(...)` | `src/hippocampai/telemetry.py` | Context manager wrapping a provider adapter's public `chat`/`generate` method; opens/finalizes one `LogicalInvocation`. |
| `llm_attempt(...)` | `src/hippocampai/telemetry.py` | Context manager wrapping one real provider request; opens/records one `ProviderAttempt`. |
| `estimate_cost(...)` | `src/hippocampai/telemetry.py` | Reads `Config.llm_pricing`; returns `None` for unpriced provider/model pairs. |
| `sanitize_error_message(...)`, `classify_llm_error(...)` | `src/hippocampai/telemetry.py` | Privacy/sanitization helpers described above. |

---

## See also

- [`TELEMETRY.md`](TELEMETRY.md#llm-usage-tracing): usage guide, code examples, configuration.
- [`scripts/validate_llm_tracing.py`](../scripts/validate_llm_tracing.py): runnable, offline, bounded validation scenario producing exactly the parent/child example shown in this document.
- [`examples/traces/provider_trace_example.json`](../examples/traces/provider_trace_example.json): the sanitized JSON example, as a standalone fixture.
- [`tests/test_llm_telemetry.py`](../tests/test_llm_telemetry.py): 34 tests covering every behavior documented here.
