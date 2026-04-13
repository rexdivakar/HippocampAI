#!/usr/bin/env python3
"""
HippocampAI — End-to-End Test Suite
====================================
Tests every live component one by one, sequentially.
No existing test scripts reused. No concurrent requests.

Stack assumed running at:
  API        → http://localhost:8000
  Qdrant     → http://localhost:6333
  Redis      → localhost:6379
  Prometheus → http://localhost:9090
  Grafana    → http://localhost:3000
  Flower     → http://localhost:5555
  Frontend   → http://localhost:81
  Admin UI   → http://localhost:3001

Run:
    python scripts/e2e_test.py
"""

import sys
import time
import uuid
from typing import Any

import httpx

# ─── Configuration ──────────────────────────────────────────────────────────
BASE        = "http://localhost:8000"
QDRANT      = "http://localhost:6333"
PROMETHEUS  = "http://localhost:9090"
GRAFANA     = "http://localhost:3000"
FLOWER      = "http://localhost:5555"
FRONTEND    = "http://localhost:81"
ADMIN_UI    = "http://localhost:3001"
REDIS_PORT  = 6379
TIMEOUT     = 30

USER_ID = f"e2e_{uuid.uuid4().hex[:8]}"
PASS = "✅ PASS"
FAIL = "❌ FAIL"

results: list[dict[str, Any]] = []


# ─── Helpers ────────────────────────────────────────────────────────────────

def section(title: str) -> None:
    print(f"\n{'═'*62}")
    print(f"  {title}")
    print(f"{'═'*62}")


def check(name: str, ok: bool, detail: str = "") -> None:
    icon = PASS if ok else FAIL
    print(f"  {icon}  {name}")
    if detail:
        print(f"           {detail}")
    results.append({"name": name, "ok": ok, "detail": detail})


def safe_get(url: str, params: dict | None = None, timeout: int = TIMEOUT) -> tuple[int, Any]:
    try:
        r = httpx.get(url, params=params, timeout=timeout)
        try:
            return r.status_code, r.json()
        except Exception:
            return r.status_code, r.text
    except Exception as exc:
        return 0, str(exc)


def safe_post(url: str, body: dict | None = None, params: dict | None = None,
              timeout: int = TIMEOUT) -> tuple[int, Any]:
    try:
        r = httpx.post(url, json=body, params=params, timeout=timeout)
        try:
            return r.status_code, r.json()
        except Exception:
            return r.status_code, r.text
    except Exception as exc:
        return 0, str(exc)


def safe_patch(url: str, body: dict | None = None) -> tuple[int, Any]:
    try:
        r = httpx.patch(url, json=body, timeout=TIMEOUT)
        try:
            return r.status_code, r.json()
        except Exception:
            return r.status_code, r.text
    except Exception as exc:
        return 0, str(exc)


def safe_delete(url: str, body: dict | None = None,
                params: dict | None = None) -> tuple[int, Any]:
    try:
        r = httpx.request("DELETE", url, json=body, params=params, timeout=TIMEOUT)
        try:
            return r.status_code, r.json()
        except Exception:
            return r.status_code, r.text
    except Exception as exc:
        return 0, str(exc)


def api(path: str) -> str:
    return f"{BASE}{path}"


# ════════════════════════════════════════════════════════════════════════════
# 1. Infrastructure health
# ════════════════════════════════════════════════════════════════════════════
section("1. INFRASTRUCTURE HEALTH")

code, body = safe_get(api("/healthz"))
check("API /healthz → 200", code == 200,
      str(body)[:80])

code, body = safe_get(api("/health"))
check("API /health → 200", code == 200,
      str(body)[:80])

code, body = safe_get(api("/metrics"))
check("API /metrics (Prometheus scrape) → 200",
      code == 200, f"{len(str(body))} bytes")

# Qdrant — root returns version JSON, /health returns 404 in v1.15
code, body = safe_get(f"{QDRANT}/")
check("Qdrant root reachable",
      code == 200 and "version" in str(body),
      str(body)[:80])

code, body = safe_get(f"{QDRANT}/collections")
collections = [c["name"] for c in (body.get("result", {}).get("collections", []) if isinstance(body, dict) else [])]
check("Qdrant /collections → 200",
      code == 200, f"collections={collections}")

try:
    import redis as redis_lib
    rc = redis_lib.Redis(host="localhost", port=REDIS_PORT, db=0, socket_timeout=5)
    pong = rc.ping()
    check("Redis PING", pong is True, "PONG")
except Exception as exc:
    check("Redis PING", False, str(exc))

code, body = safe_get(f"{PROMETHEUS}/-/healthy")
check("Prometheus /-/healthy → 200", code == 200, str(body)[:60])

code, body = safe_get(f"{GRAFANA}/api/health")
check("Grafana /api/health → 200",
      code == 200,
      str(body)[:80])

code, body = safe_get(f"{FLOWER}/healthcheck")
check("Flower /healthcheck → 200", code == 200, str(body)[:60])

code, body = safe_get(FRONTEND)
check("Frontend (React SPA) → 200", code == 200)

code, body = safe_get(ADMIN_UI)
check("Admin UI (nginx) → 200", code == 200)


# ════════════════════════════════════════════════════════════════════════════
# 2. Memory classification
# ════════════════════════════════════════════════════════════════════════════
section("2. MEMORY CLASSIFICATION")

classify_cases = [
    ("I always prefer dark mode in every application I use", "preference"),
    ("The capital of France is Paris", "fact"),
    ("My goal is to run a full marathon before turning 35", "goal"),
    ("I drink green tea every morning before opening my laptop", "habit"),
    ("I attended the all-hands meeting yesterday afternoon", "event"),
]

for text, expected in classify_cases:
    code, body = safe_post(api("/v1/classify"), {"text": text})
    if code == 200 and isinstance(body, dict):
        detected = body.get("memory_type", "")
        ok = detected == expected
        detail = f"expected={expected}  got={detected}  conf={body.get('confidence',0):.2f}"
    else:
        ok = False
        detail = f"HTTP {code}: {str(body)[:80]}"
    check(f"Classify → '{text[:48]}'", ok, detail)


# ════════════════════════════════════════════════════════════════════════════
# 3. Remember (create memories)
# ════════════════════════════════════════════════════════════════════════════
section("3. REMEMBER — CREATE MEMORIES")

seeds = [
    {"text": "I love hiking in the mountains every weekend with my dog",
     "memory_type": "habit", "importance": 8.0, "tags": ["outdoor", "fitness", "dog"]},
    {"text": "Python is my primary language for machine learning projects",
     "memory_type": "fact", "importance": 7.5, "tags": ["python", "ml"]},
    {"text": "My goal is to become a principal engineer within 3 years",
     "memory_type": "goal", "importance": 9.5, "tags": ["career", "growth"]},
    {"text": "I attended the Q2 engineering all-hands and took detailed notes",
     "memory_type": "event", "importance": 6.0, "tags": ["work", "meeting"]},
    {"text": "I prefer concise code reviews with specific line-level feedback",
     "memory_type": "preference", "importance": 8.0, "tags": ["work", "code-review"]},
    {"text": "The team ships to production every Thursday using GitHub Actions CI",
     "memory_type": "fact", "importance": 6.5, "tags": ["work", "devops"]},
    {"text": "I struggle with work-life balance when deadlines are tight",
     "memory_type": "context", "importance": 7.0, "tags": ["wellbeing"]},
    {"text": "I completed the Rust programming course on Udemy last month",
     "memory_type": "event", "importance": 7.5, "tags": ["rust", "learning"]},
]

created_ids: list[str] = []
for mem in seeds:
    payload = {"user_id": USER_ID, **mem}
    code, body = safe_post(api("/v1/memories:remember"), payload)
    ok = code == 200 and isinstance(body, dict) and "id" in body
    if ok:
        mid = body["id"]
        created_ids.append(mid)
        detail = f"id={mid[:16]}  type={body.get('type')}  importance={body.get('importance')}"
    else:
        detail = f"HTTP {code}: {str(body)[:100]}"
    check(f"Remember: '{mem['text'][:48]}'", ok, detail)
    time.sleep(0.25)

print(f"\n  → {len(created_ids)}/{len(seeds)} memories created for user {USER_ID}")


# ════════════════════════════════════════════════════════════════════════════
# 4. Get memory by ID
# ════════════════════════════════════════════════════════════════════════════
section("4. GET MEMORY BY ID")

if created_ids:
    target = created_ids[0]
    code, body = safe_get(api(f"/v1/memories/{target}"))
    ok = code == 200 and isinstance(body, dict) and body.get("id") == target
    detail = f"id={body.get('id','')[:16]}  text='{body.get('text','')[:40]}'" if ok else f"HTTP {code}: {str(body)[:80]}"
    check("Get memory by ID (first created)", ok, detail)

    # Non-existent ID
    fake_id = str(uuid.uuid4())
    code, body = safe_get(api(f"/v1/memories/{fake_id}"))
    check("Non-existent memory returns 404", code == 404, f"HTTP {code}")


# ════════════════════════════════════════════════════════════════════════════
# 5. Get all memories for user
# ════════════════════════════════════════════════════════════════════════════
section("5. GET ALL USER MEMORIES")

code, body = safe_post(api("/v1/memories:get"),
                       {"user_id": USER_ID, "limit": 50})
ok = code == 200 and isinstance(body, list)
count = len(body) if isinstance(body, list) else 0
check("Get all memories for user", ok and count >= len(created_ids),
      f"returned {count} memories (created {len(created_ids)})")

# Filter by type
code, body = safe_post(api("/v1/memories:get"),
                       {"user_id": USER_ID, "filters": {"type": "habit"}, "limit": 10})
ok = code == 200 and isinstance(body, list)
habit_types = [m.get("type") for m in body] if isinstance(body, list) else []
check("Get memories filtered by type=habit",
      ok and all(t == "habit" for t in habit_types),
      f"returned {len(habit_types)} habit memories")


# ════════════════════════════════════════════════════════════════════════════
# 6. Update memory
# ════════════════════════════════════════════════════════════════════════════
section("6. UPDATE MEMORY")

if len(created_ids) >= 2:
    target = created_ids[1]
    code, body = safe_patch(api("/v1/memories:update"),
                            {"memory_id": target, "user_id": USER_ID,
                             "importance": 9.9,
                             "tags": ["python", "ml", "updated"]})
    ok = code == 200 and isinstance(body, dict) and body.get("importance") == 9.9
    detail = (f"importance={body.get('importance')}  tags={body.get('tags')}"
              if isinstance(body, dict) else f"HTTP {code}: {str(body)[:80]}")
    check("Update memory importance to 9.9 and add tag", ok, detail)


# ════════════════════════════════════════════════════════════════════════════
# 7. Recall — hybrid search
# ════════════════════════════════════════════════════════════════════════════
section("7. RECALL — HYBRID SEARCH")

recall_cases = [
    ("outdoor weekend activities with dog",   ["hiking", "dog"],        "habit retrieval"),
    ("programming language for ML",           ["python", "machine"],    "fact retrieval"),
    ("career development and promotion",      ["principal", "engineer"],"goal retrieval"),
    ("how I prefer code feedback",            ["concise", "feedback"],  "preference retrieval"),
    ("production deployment pipeline",        ["GitHub", "Thursday"],   "fact + devops"),
    ("what courses have I completed",         ["Rust", "Udemy"],        "event retrieval"),
]

for query, keywords, desc in recall_cases:
    code, body = safe_post(api("/v1/memories:recall"),
                           {"user_id": USER_ID, "query": query, "k": 5})
    if code == 200 and isinstance(body, list):
        all_text = " ".join(
            r.get("memory", {}).get("text", "").lower() for r in body
        )
        found = [kw for kw in keywords if kw.lower() in all_text]
        ok = len(found) > 0
        detail = (f"returned={len(body)}  found_keywords={found}  scores="
                  f"{[round(r.get('score',0),3) for r in body[:3]]}  [{desc}]")
    else:
        ok = False
        detail = f"HTTP {code}: {str(body)[:80]}"
    check(f"Recall: '{query}'", ok, detail)
    time.sleep(0.4)


# ════════════════════════════════════════════════════════════════════════════
# 8. Deduplication
# ════════════════════════════════════════════════════════════════════════════
section("8. DEDUPLICATION")

# Store a near-duplicate — should return the existing memory
dup_text = "I love hiking in the mountains every weekend with my dog"
code, body = safe_post(api("/v1/memories:remember"),
                       {"user_id": USER_ID, "text": dup_text,
                        "memory_type": "habit", "importance": 8.0})
if code == 200 and isinstance(body, dict):
    returned_id = body.get("id", "")
    is_existing = returned_id in created_ids
    check("Near-duplicate remember returns existing memory",
          is_existing,
          f"returned_id={'EXISTING (' + returned_id[:12] + ')' if is_existing else 'NEW: '+returned_id[:12]}")
else:
    check("Near-duplicate remember call", False, f"HTTP {code}: {str(body)[:80]}")

# Explicit deduplication scan
code, body = safe_post(api("/v1/memories/deduplicate"),
                       {"user_id": USER_ID, "similarity_threshold": 0.85})
check("POST /v1/memories/deduplicate",
      code == 200, str(body)[:120])


# ════════════════════════════════════════════════════════════════════════════
# 9. Batch operations
# ════════════════════════════════════════════════════════════════════════════
section("9. BATCH OPERATIONS")

batch_mems = [
    {"text": "I use Neovim as my daily driver editor",
     "memory_type": "preference", "importance": 6.5},
    {"text": "I passed the AWS Solutions Architect exam",
     "memory_type": "event", "importance": 8.5},
    {"text": "My favourite database is PostgreSQL for transactional workloads",
     "memory_type": "preference", "importance": 7.0},
]
code, body = safe_post(api("/v1/memories/batch"),
                       {"user_id": USER_ID, "memories": batch_mems})
batch_ids: list[str] = []
if code == 200 and isinstance(body, list):
    batch_ids = [m["id"] for m in body if "id" in m]
    ok = len(batch_ids) == len(batch_mems)
    detail = f"created {len(batch_ids)}/{len(batch_mems)}"
else:
    ok = False
    detail = f"HTTP {code}: {str(body)[:100]}"
check("Batch remember (3 memories)", ok, detail)

if batch_ids:
    code, body = safe_post(api("/v1/memories/batch/get"),
                           {"user_id": USER_ID, "memory_ids": batch_ids})
    ok = code == 200 and isinstance(body, list) and len(body) == len(batch_ids)
    check("Batch get by IDs",
          ok, f"retrieved {len(body) if isinstance(body, list) else 0}/{len(batch_ids)}")

    code, body = safe_post(api("/v1/memories/batch/delete"),
                           {"user_id": USER_ID, "memory_ids": [batch_ids[-1]]})
    check("Batch delete (last batch memory)",
          code == 200, f"HTTP {code}: {str(body)[:60]}")


# ════════════════════════════════════════════════════════════════════════════
# 10. Delete memory
# ════════════════════════════════════════════════════════════════════════════
section("10. DELETE SINGLE MEMORY")

if len(created_ids) >= 3:
    del_id = created_ids.pop()
    code, body = safe_delete(api("/v1/memories:delete"),
                             {"memory_id": del_id, "user_id": USER_ID})
    check("Delete memory by ID",
          code == 200, f"HTTP {code}: {str(body)[:60]}")

    # Verify gone
    code, _ = safe_get(api(f"/v1/memories/{del_id}"))
    check("Deleted memory is no longer retrievable",
          code == 404, f"got HTTP {code}")


# ════════════════════════════════════════════════════════════════════════════
# 11. Memory extraction from conversation
# ════════════════════════════════════════════════════════════════════════════
section("11. EXTRACT MEMORIES FROM CONVERSATION")

convo = (
    "Just finished implementing OAuth2 login with PKCE. "
    "I really enjoy working with FastAPI for building secure APIs. "
    "The feature went live at 5pm and the error rate stayed below 0.1%."
)
code, body = safe_post(api("/v1/memories:extract"),
                       {"user_id": USER_ID, "conversation": convo})
ok = code == 200 and isinstance(body, list)
detail = (f"extracted {len(body)} memories → types={[m.get('type') for m in body]}"
          if isinstance(body, list) else f"HTTP {code}: {str(body)[:100]}")
check("Extract memories from conversation text", ok, detail)


# ════════════════════════════════════════════════════════════════════════════
# 12. Relevance feedback
# ════════════════════════════════════════════════════════════════════════════
section("12. RELEVANCE FEEDBACK")

if created_ids:
    fid = created_ids[0]
    code, body = safe_post(api(f"/v1/memories/{fid}/feedback"),
                           {"user_id": USER_ID, "query": "outdoor hiking activities",
                            "feedback_type": "relevant", "score": 1.0})
    check("Submit positive feedback on memory",
          code in (200, 201), f"HTTP {code}: {str(body)[:80]}")

    code, body = safe_post(api(f"/v1/memories/{created_ids[1]}/feedback"),
                           {"user_id": USER_ID, "query": "python machine learning",
                            "feedback_type": "not_relevant", "score": -0.5})
    check("Submit negative feedback on memory",
          code in (200, 201), f"HTTP {code}: {str(body)[:80]}")

code, body = safe_get(api("/v1/feedback/stats"),
                      params={"user_id": USER_ID})
check("GET /v1/feedback/stats",
      code == 200, str(body)[:120])


# ════════════════════════════════════════════════════════════════════════════
# 13. Memory triggers
# ════════════════════════════════════════════════════════════════════════════
section("13. MEMORY TRIGGERS")

code, body = safe_get(api("/v1/triggers"), params={"user_id": USER_ID})
check("GET /v1/triggers (list)", code == 200, str(body)[:80])

trigger_payload = {
    "user_id": USER_ID,
    "name": "e2e_trigger",
    "event": "on_remember",
    "conditions": [{"field": "memory_type", "op": "eq", "value": "goal"}],
    "action": "log",
    "action_config": {"message": "New goal stored"},
}
code, body = safe_post(api("/v1/triggers"), trigger_payload)
trigger_id = body.get("id", "") if isinstance(body, dict) else ""
check("POST /v1/triggers (create)",
      code in (200, 201), f"id={trigger_id[:16] if trigger_id else '?'}  HTTP {code}")

if trigger_id:
    # Verify trigger appears in list (no GET-by-ID route exists)
    code, body = safe_get(api("/v1/triggers"), params={"user_id": USER_ID})
    ids_in_list = [t.get("id") for t in (body if isinstance(body, list) else [])]
    check("GET /v1/triggers includes new trigger",
          trigger_id in ids_in_list, str(body)[:80])

    code, body = safe_get(api(f"/v1/triggers/{trigger_id}/history"))
    check("GET /v1/triggers/{id}/history", code == 200, str(body)[:80])

    code, body = safe_delete(api(f"/v1/triggers/{trigger_id}"),
                             params={"user_id": USER_ID})
    check("DELETE /v1/triggers/{id}", code in (200, 204), f"HTTP {code}")


# ════════════════════════════════════════════════════════════════════════════
# 14. Sleep phase / consolidation
# ════════════════════════════════════════════════════════════════════════════
section("14. SLEEP PHASE / CONSOLIDATION")

code, body = safe_get(api("/api/consolidation/status"))
check("GET /api/consolidation/status",
      code == 200, str(body)[:100])

code, body = safe_get(api("/api/consolidation/stats"))
check("GET /api/consolidation/stats",
      code == 200, str(body)[:100])

code, body = safe_post(api("/api/consolidation/trigger"),
                       params={"user_id": USER_ID},
                       body={"dry_run": True, "lookback_hours": 1},
                       timeout=90)
check("POST /api/consolidation/trigger (dry_run)",
      code in (200, 202), str(body)[:150])

code, body = safe_get(api("/api/consolidation/runs"))
check("GET /api/consolidation/runs",
      code == 200, str(body)[:100])


# ════════════════════════════════════════════════════════════════════════════
# 15. Bi-temporal memory
# ════════════════════════════════════════════════════════════════════════════
section("15. BI-TEMPORAL MEMORY")

bt_payload = {
    "text": "Python latest stable version is 3.13",
    "user_id": USER_ID,
    "property_name": "latest_version",
    "valid_from": "2024-10-01T00:00:00Z",
    "source": "e2e_test",
    "confidence": 0.99,
}
code, body = safe_post(api("/v1/bitemporal/facts:store"), bt_payload)
bt_id = body.get("id", "") if isinstance(body, dict) else ""
check("POST /v1/bitemporal/facts:store",
      code in (200, 201),
      f"id={bt_id}  HTTP {code}: {str(body)[:80]}")

code, body = safe_post(api("/v1/bitemporal/facts:query"),
                       {"user_id": USER_ID, "text": "Python version"})
check("POST /v1/bitemporal/facts:query",
      code == 200, str(body)[:120])

code, body = safe_post(api("/v1/bitemporal/facts:history"),
                       {"user_id": USER_ID, "fact_id": bt_id})
check("POST /v1/bitemporal/facts:history",
      code == 200, str(body)[:120])


# ════════════════════════════════════════════════════════════════════════════
# 16. Context assembly
# ════════════════════════════════════════════════════════════════════════════
section("16. CONTEXT ASSEMBLY")

code, body = safe_post(api("/v1/context:assemble"),
                       {"user_id": USER_ID,
                        "query": "What are my programming preferences and career goals?",
                        "max_tokens": 1024})
ok = code == 200
detail = (f"tokens_used={body.get('tokens_used',0)}  "
          f"memories_used={len(body.get('memories',[]))}"
          if isinstance(body, dict) else f"HTTP {code}: {str(body)[:100]}")
check("POST /v1/context:assemble (JSON context)", ok, detail)

code, body = safe_post(api("/v1/context:assemble/text"),
                       {"user_id": USER_ID,
                        "query": "Tell me about my work habits",
                        "max_tokens": 512})
check("POST /v1/context:assemble/text (plain-text context)",
      code == 200, str(body)[:120])


# ════════════════════════════════════════════════════════════════════════════
# 17. Procedural memory
# ════════════════════════════════════════════════════════════════════════════
section("17. PROCEDURAL MEMORY")

_PROC_DISABLED_MSG = "Procedural memory is disabled"

code, body = safe_get(api("/v1/procedural/rules"),
                      params={"user_id": USER_ID})
_proc_disabled = isinstance(body, dict) and _PROC_DISABLED_MSG in str(body)
check("GET /v1/procedural/rules",
      code == 200 or _proc_disabled,
      str(body)[:100] if not _proc_disabled else "[SKIP — set ENABLE_PROCEDURAL_MEMORY=true]")

extract_payload = {
    "user_id": USER_ID,
    "interactions": [
        {"role": "user", "content": "How do I deploy to production?"},
        {"role": "assistant",
         "content": "Run tests, build Docker image, push to registry, deploy to staging, verify, then promote to prod."},
        {"role": "user", "content": "When should I skip staging?"},
        {"role": "assistant", "content": "Never skip staging for user-facing changes."},
    ],
}
code, body = safe_post(api("/v1/procedural/extract"), extract_payload)
check("POST /v1/procedural/extract (from interactions)",
      code in (200, 201) or _proc_disabled,
      str(body)[:150] if not _proc_disabled else "[SKIP — procedural memory disabled]")

inject_payload = {
    "user_id": USER_ID,
    "base_prompt": "How should I deploy this feature? We have a new authentication feature ready.",
}
code, body = safe_post(api("/v1/procedural/inject"), inject_payload)
check("POST /v1/procedural/inject (inject rules into context)",
      code == 200 or _proc_disabled,
      str(body)[:120] if not _proc_disabled else "[SKIP — procedural memory disabled]")


# ════════════════════════════════════════════════════════════════════════════
# 18. Prospective memory
# ════════════════════════════════════════════════════════════════════════════
section("18. PROSPECTIVE MEMORY")

_PROSP_DISABLED_MSG = "Prospective memory is disabled"

code, body = safe_get(api("/v1/prospective/intents"),
                      params={"user_id": USER_ID})
_prosp_disabled = isinstance(body, dict) and _PROSP_DISABLED_MSG in str(body)
check("GET /v1/prospective/intents (list)",
      code == 200 or _prosp_disabled,
      str(body)[:100] if not _prosp_disabled else "[SKIP — set ENABLE_PROSPECTIVE_MEMORY=true]")

parse_payload = {
    "user_id": USER_ID,
    "text": "Remind me to review the architecture doc before the Friday sprint review",
}
code, body = safe_post(api("/v1/prospective/intents:parse"), parse_payload)
check("POST /v1/prospective/intents:parse",
      code in (200, 201) or _prosp_disabled,
      str(body)[:150] if not _prosp_disabled else "[SKIP — prospective memory disabled]")

intent_payload = {
    "user_id": USER_ID,
    "intent_text": "Review Q3 OKRs before the board meeting",
    "trigger_type": "time",
    "trigger_at": "2026-06-30T09:00:00Z",
    "priority": 2,
    "tags": ["work", "okr"],
}
code, body = safe_post(api("/v1/prospective/intents"), intent_payload)
intent_id = body.get("id", "") if isinstance(body, dict) else ""
check("POST /v1/prospective/intents (create intent)",
      code in (200, 201) or _prosp_disabled,
      f"id={intent_id}  HTTP {code}" if not _prosp_disabled else "[SKIP — prospective memory disabled]")

if intent_id and not _prosp_disabled:
    code, body = safe_get(api(f"/v1/prospective/intents/{intent_id}"))
    check("GET /v1/prospective/intents/{id}",
          code == 200, str(body)[:80])

    code, body = safe_post(api(f"/v1/prospective/intents/{intent_id}/complete"),
                           {"user_id": USER_ID})
    check("POST /v1/prospective/intents/{id}/complete",
          code == 200, str(body)[:80])

code, body = safe_post(api("/v1/prospective/evaluate"),
                       {"user_id": USER_ID, "context_text": "About to attend board meeting"})
check("POST /v1/prospective/evaluate",
      code == 200 or _prosp_disabled,
      str(body)[:120] if not _prosp_disabled else "[SKIP — prospective memory disabled]")


# ════════════════════════════════════════════════════════════════════════════
# 19. Intelligence — entities, relationships, clustering
# ════════════════════════════════════════════════════════════════════════════
section("19. INTELLIGENCE — ENTITIES & RELATIONSHIPS")

code, body = safe_post(api("/v1/intelligence/entities:extract"),
                       {"user_id": USER_ID, "text": convo})
check("POST /v1/intelligence/entities:extract",
      code == 200, str(body)[:150])

code, body = safe_post(api("/v1/intelligence/entities:search"),
                       {"user_id": USER_ID, "query": "Python"})
check("POST /v1/intelligence/entities:search",
      code == 200, str(body)[:150])

code, body = safe_post(api("/v1/intelligence/relationships:analyze"),
                       {"user_id": USER_ID, "text": convo})
check("POST /v1/intelligence/relationships:analyze",
      code in (200, 202), str(body)[:120])

_mem_dicts = [
    {"text": s["text"], "type": s["memory_type"], "user_id": USER_ID}
    for s in seeds
]
code, body = safe_post(api("/v1/intelligence/clustering:analyze"),
                       {"user_id": USER_ID, "memories": _mem_dicts})
check("POST /v1/intelligence/clustering:analyze",
      code in (200, 202), str(body)[:120])

code, body = safe_post(api("/v1/intelligence/temporal:analyze"),
                       {"user_id": USER_ID, "memories": _mem_dicts,
                        "analysis_type": "trends"})
check("POST /v1/intelligence/temporal:analyze",
      code in (200, 202), str(body)[:120])

code, body = safe_get(api("/v1/intelligence/health"))
check("GET /v1/intelligence/health",
      code == 200, str(body)[:80])


# ════════════════════════════════════════════════════════════════════════════
# 20. Self-healing
# ════════════════════════════════════════════════════════════════════════════
section("20. SELF-HEALING")

code, body = safe_post(api("/v1/healing/health"),
                       {"user_id": USER_ID, "detailed": True})
check("POST /v1/healing/health (full check)",
      code == 200, str(body)[:150])

code, body = safe_post(api("/v1/healing/deduplication"),
                       {"user_id": USER_ID})
check("POST /v1/healing/deduplication",
      code in (200, 202), str(body)[:120])

code, body = safe_post(api("/v1/healing/importance"),
                       {"user_id": USER_ID})
check("POST /v1/healing/importance (recalculate scores)",
      code in (200, 202), str(body)[:120])

code, body = safe_post(api("/v1/healing/tagging"),
                       {"user_id": USER_ID})
check("POST /v1/healing/tagging (auto-tag)",
      code in (200, 202), str(body)[:120])

code, body = safe_post(api("/v1/healing/health/duplicates"),
                       params={"user_id": USER_ID})
check("POST /v1/healing/health/duplicates",
      code == 200, str(body)[:120])

code, body = safe_post(api("/v1/healing/health/stale"),
                       params={"user_id": USER_ID})
check("POST /v1/healing/health/stale",
      code == 200, str(body)[:120])

code, body = safe_post(api("/v1/healing/health/gaps"),
                       params={"user_id": USER_ID})
check("POST /v1/healing/health/gaps",
      code == 200, str(body)[:120])


# ════════════════════════════════════════════════════════════════════════════
# 21. Predictions / analytics
# ════════════════════════════════════════════════════════════════════════════
section("21. PREDICTIONS & ANALYTICS")

code, body = safe_post(api("/v1/predictions/trends"),
                       params={"user_id": USER_ID})
check("POST /v1/predictions/trends",
      code == 200, str(body)[:120])

code, body = safe_post(api("/v1/predictions/anomalies"),
                       {"user_id": USER_ID})
check("POST /v1/predictions/anomalies",
      code == 200, str(body)[:120])

code, body = safe_post(api("/v1/predictions/insights"),
                       {"user_id": USER_ID})
check("POST /v1/predictions/insights",
      code == 200, str(body)[:120])

code, body = safe_post(api("/v1/predictions/recommendations"),
                       {"user_id": USER_ID})
check("POST /v1/predictions/recommendations",
      code == 200, str(body)[:120])

code, body = safe_post(api("/v1/predictions/activity/peaks"),
                       params={"user_id": USER_ID})
check("POST /v1/predictions/activity/peaks",
      code == 200, str(body)[:120])


# ════════════════════════════════════════════════════════════════════════════
# 22. Collaboration spaces
# ════════════════════════════════════════════════════════════════════════════
section("22. COLLABORATION SPACES")

space_payload = {
    "name": "e2e_collab_space",
    "owner_agent_id": USER_ID,
    "description": "E2E test collaboration space",
    "tags": ["e2e", "test"],
}
code, body = safe_post(api("/v1/collaboration/spaces"), space_payload)
space_id = body.get("id", "") if isinstance(body, dict) else ""
check("POST /v1/collaboration/spaces (create space)",
      code in (200, 201),
      f"id={space_id}  HTTP {code}")

if space_id:
    code, body = safe_get(api(f"/v1/collaboration/spaces/{space_id}"))
    check("GET /v1/collaboration/spaces/{id}",
          code == 200, str(body)[:80])

    code, body = safe_get(api(f"/v1/collaboration/spaces/{space_id}/events"))
    check("GET /v1/collaboration/spaces/{id}/events",
          code == 200, str(body)[:80])

    code, body = safe_get(api(f"/v1/collaboration/spaces/{space_id}/memories"))
    check("GET /v1/collaboration/spaces/{id}/memories",
          code == 200, str(body)[:80])

code, body = safe_get(api("/v1/collaboration/conflicts"))
check("GET /v1/collaboration/conflicts",
      code == 200, str(body)[:80])


# ════════════════════════════════════════════════════════════════════════════
# 23. Sessions
# ════════════════════════════════════════════════════════════════════════════
section("23. SESSIONS")

code, body = safe_get(api("/api/sessions/list"),
                      params={"user_id": USER_ID})
check("GET /api/sessions/list", code == 200, str(body)[:100])

code, body = safe_get(api("/api/sessions/stats"),
                      params={"user_id": USER_ID})
check("GET /api/sessions/stats", code == 200, str(body)[:100])


# ════════════════════════════════════════════════════════════════════════════
# 24. Memory compaction
# ════════════════════════════════════════════════════════════════════════════
section("24. MEMORY COMPACTION")

code, body = safe_get(api("/api/compaction/types"))
check("GET /api/compaction/types", code == 200, str(body)[:120])

code, body = safe_get(api("/api/compaction/preview"),
                      params={"user_id": USER_ID})
check("GET /api/compaction/preview",
      code == 200, str(body)[:120])


# ════════════════════════════════════════════════════════════════════════════
# 25. Dashboard stats
# ════════════════════════════════════════════════════════════════════════════
section("25. DASHBOARD STATS")

code, body = safe_get(api("/api/dashboard/stats"),
                      params={"user_id": USER_ID})
check("GET /api/dashboard/stats",
      code == 200, str(body)[:150])

code, body = safe_get(api("/api/dashboard/recent-activity"),
                      params={"user_id": USER_ID})
check("GET /api/dashboard/recent-activity",
      code == 200, str(body)[:100])


# ════════════════════════════════════════════════════════════════════════════
# 26. Expire memories
# ════════════════════════════════════════════════════════════════════════════
section("26. MEMORY EXPIRY")

code, body = safe_post(api("/v1/memories:remember"),
                       {"user_id": USER_ID,
                        "text": "Temporary context note — expires immediately",
                        "memory_type": "context",
                        "importance": 1.0,
                        "ttl_days": 0})
ok = code == 200 and isinstance(body, dict)
exp_id = body.get("id", "") if ok else ""
check("Remember memory with ttl_days=0", ok,
      f"id={exp_id[:16] if exp_id else '?'}")

if exp_id:
    time.sleep(0.5)
    code, body = safe_post(api("/v1/memories:expire"),
                           {"user_id": USER_ID})
    check("POST /v1/memories:expire",
          code == 200, str(body)[:80])


# ════════════════════════════════════════════════════════════════════════════
# 27. Celery / Flower worker status
# ════════════════════════════════════════════════════════════════════════════
section("27. CELERY / FLOWER")

try:
    r = httpx.get(f"{FLOWER}/api/workers", timeout=10, auth=("admin", "changeme_in_production"))
    if r.status_code == 200:
        workers = r.json()
        names = list(workers.keys())
        check("Flower: workers registered",
              len(names) > 0, f"{len(names)} worker(s): {names[:2]}")
    else:
        check("Flower /api/workers", False, f"HTTP {r.status_code}")
except Exception as exc:
    check("Flower /api/workers", False, str(exc))

try:
    r = httpx.get(f"{FLOWER}/api/tasks", timeout=10, auth=("admin", "changeme_in_production"))
    check("Flower /api/tasks accessible", r.status_code == 200, f"HTTP {r.status_code}")
except Exception as exc:
    check("Flower /api/tasks", False, str(exc))


# ════════════════════════════════════════════════════════════════════════════
# 28. Prometheus metrics populated
# ════════════════════════════════════════════════════════════════════════════
section("28. PROMETHEUS METRICS")

metrics_to_check = [
    "memories_created_total",
    "memory_operations_total",
    "search_requests_total",
    "process_resident_memory_bytes",
]
for metric in metrics_to_check:
    try:
        r = httpx.get(f"{PROMETHEUS}/api/v1/query",
                      params={"query": metric}, timeout=10)
        if r.status_code == 200:
            series = r.json().get("data", {}).get("result", [])
            check(f"Prometheus metric: {metric}",
                  len(series) > 0,
                  f"{len(series)} series")
        else:
            check(f"Prometheus metric: {metric}", False, f"HTTP {r.status_code}")
    except Exception as exc:
        check(f"Prometheus metric: {metric}", False, str(exc))


# ════════════════════════════════════════════════════════════════════════════
# 29. Qdrant — verify test user data persisted
# ════════════════════════════════════════════════════════════════════════════
section("29. QDRANT — DATA PERSISTENCE VERIFICATION")

for coll_name in ["hippocampai_facts", "hippocampai_prefs"]:
    try:
        r = httpx.get(f"{QDRANT}/collections/{coll_name}", timeout=10)
        if r.status_code == 200:
            info = r.json().get("result", {})
            vec_count = info.get("vectors_count", 0)
            status = info.get("status", "?")
            check(f"Qdrant collection '{coll_name}'",
                  status in ("green", "yellow"),
                  f"vectors={vec_count}  status={status}")
        else:
            check(f"Qdrant collection '{coll_name}'", False, f"HTTP {r.status_code}")
    except Exception as exc:
        check(f"Qdrant collection '{coll_name}'", False, str(exc))

# Scroll to verify our test user's vectors are present
try:
    r = httpx.post(
        f"{QDRANT}/collections/hippocampai_facts/points/scroll",
        json={"filter": {"must": [{"key": "user_id", "match": {"value": USER_ID}}]},
              "limit": 20, "with_payload": True},
        timeout=10,
    )
    if r.status_code == 200:
        points = r.json().get("result", {}).get("points", [])
        check("Qdrant: test-user fact vectors persisted",
              len(points) > 0,
              f"{len(points)} points in facts collection")
    else:
        check("Qdrant: test-user fact vectors", False, f"HTTP {r.status_code}")
except Exception as exc:
    check("Qdrant: test-user fact vectors", False, str(exc))


# ════════════════════════════════════════════════════════════════════════════
# 30. Redis — BM25 dirty key and cache entries
# ════════════════════════════════════════════════════════════════════════════
section("30. REDIS — CACHE & BM25 STATE")

try:
    import redis as redis_lib
    rc = redis_lib.Redis(host="localhost", port=REDIS_PORT, db=0, socket_timeout=5)

    ts_key = f"hippocampai:bm25_write_ts:{USER_ID}"
    ts_val = rc.get(ts_key)
    check("Redis: BM25 write-timestamp key set",
          ts_val is not None,
          f"key={ts_key}  value={ts_val.decode() if ts_val else 'NOT SET'}")

    cache_keys = rc.keys("memory:*")
    check("Redis: memory cache keys present",
          len(cache_keys) > 0,
          f"{len(cache_keys)} cache entries")
except Exception as exc:
    check("Redis BM25 / cache state", False, str(exc))


# ════════════════════════════════════════════════════════════════════════════
# 31. WebSocket / Socket.IO
# ════════════════════════════════════════════════════════════════════════════
section("31. WEBSOCKET / SOCKET.IO")

try:
    import socketio as sio_lib
    sio = sio_lib.SimpleClient()
    sio.connect(BASE, wait_timeout=8)
    connected = bool(sio.sid)
    sid = sio.sid
    sio.disconnect()
    check("Socket.IO: connection established", connected, f"sid={sid}")
except Exception as exc:
    check("Socket.IO connection", False, str(exc))


# ════════════════════════════════════════════════════════════════════════════
# Final summary
# ════════════════════════════════════════════════════════════════════════════
section("FINAL SUMMARY")

passed = sum(1 for r in results if r["ok"])
failed = sum(1 for r in results if not r["ok"])
total  = len(results)
pct    = int(passed / total * 100) if total else 0

print(f"\n  Total : {total}")
print(f"  {PASS} : {passed}  ({pct}%)")
print(f"  {FAIL} : {failed}")
print(f"  Test user : {USER_ID}")

if failed:
    print("\n  ── Failed tests ──")
    for r in results:
        if not r["ok"]:
            print(f"  • {r['name']}")
            if r["detail"]:
                print(f"    {r['detail']}")

print()
sys.exit(0 if failed == 0 else 1)
