# LLM Trace Examples

`provider_trace_example.json` is a sanitized, machine-readable example of one
`LogicalInvocation` (`LLMInvocation`) with two `ProviderAttempt`
(`UpstreamAttempt`) children: one retryable failure followed by a provider
fallback that succeeds. It is the JSON export of the exact scenario run by
[`scripts/validate_llm_tracing.py`](../../scripts/validate_llm_tracing.py).

**Provenance:** generated directly from the real implementation
(`MemoryTelemetry.export_llm_invocations()`), not hand-written. Every field
name and value shape matches `src/hippocampai/telemetry.py` exactly. See
[`docs/provider_tracing.md`](../../docs/provider_tracing.md) for the full
field-by-field schema reference.

**Reading the numbers:**
- `latency_ms` values are sub-millisecond because this scenario mocks the
  provider calls (no real network I/O) to stay offline and free of API
  keys. In production these reflect real upstream request time.
- `estimated_cost` is non-null here because the fixture was generated with
  `LLM_PRICING='{"openai:gpt-4o-mini": {"input": 0.15, "output": 0.6}}'`
  set; by default (no pricing configured) this field is `null`, not `0`.
  HippocampAI never fabricates a cost for an unpriced model.

**Sanitization:** all identifiers (`tenant_id`, `user_id`, `workflow_id`,
etc.) are fictional. The `error_message` on attempt 1 demonstrates that a
secret embedded in a raw provider error (`Authorization: Bearer ...`) is
redacted to `[REDACTED]` before being stored. No prompts, completions, API
keys, or real hostnames appear anywhere in the trace.

Regenerate with:
```bash
python scripts/validate_llm_tracing.py
```
