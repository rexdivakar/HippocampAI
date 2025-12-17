"""Celery tasks for memory consolidation (Sleep Phase)."""

import json
import logging
import os
import time
import traceback
from datetime import datetime, timedelta, timezone
from typing import Any

from celery import group

from hippocampai.celery_app import celery_app
from hippocampai.consolidation.models import (
    ConsolidationRun,
    ConsolidationStatus,
    MemoryCluster,
)
from hippocampai.consolidation.policy import (
    ConsolidationPolicy,
    apply_consolidation_decisions,
)
from hippocampai.consolidation.prompts import (
    build_consolidation_prompt,
)
from hippocampai.models import Memory

logger = logging.getLogger(__name__)

# Configuration from environment
ACTIVE_CONSOLIDATION_ENABLED = os.getenv("ACTIVE_CONSOLIDATION_ENABLED", "false").lower() == "true"
CONSOLIDATION_DRY_RUN = os.getenv("CONSOLIDATION_DRY_RUN", "false").lower() == "true"
CONSOLIDATION_LOOKBACK_HOURS = int(os.getenv("CONSOLIDATION_LOOKBACK_HOURS", "24"))
CONSOLIDATION_LLM_MODEL = os.getenv("CONSOLIDATION_LLM_MODEL", "gpt-4-turbo-preview")
CONSOLIDATION_LLM_TEMPERATURE = float(os.getenv("CONSOLIDATION_LLM_TEMPERATURE", "0.3"))
CONSOLIDATION_BATCH_SIZE = int(os.getenv("CONSOLIDATION_BATCH_SIZE", "50"))

# Metrics (optional Prometheus integration)
try:
    from prometheus_client import Counter, Histogram

    consolidation_runs_total = Counter(
        "consolidation_runs_total",
        "Total consolidation runs",
        ["user_id", "status"],
    )

    consolidation_duration_seconds = Histogram(
        "consolidation_duration_seconds",
        "Consolidation run duration in seconds",
        ["user_id"],
    )

    consolidation_memories_processed = Counter(
        "consolidation_memories_processed_total",
        "Total memories processed",
        ["user_id", "action"],
    )

    METRICS_ENABLED = True
except ImportError:
    logger.warning("Prometheus client not available, metrics disabled")
    METRICS_ENABLED = False


def emit_metric_run(user_id: str, status: str):
    """Emit consolidation run metric."""
    if METRICS_ENABLED:
        consolidation_runs_total.labels(user_id=user_id, status=status).inc()


def emit_metric_duration(user_id: str, duration: float):
    """Emit consolidation duration metric."""
    if METRICS_ENABLED:
        consolidation_duration_seconds.labels(user_id=user_id).observe(duration)


def emit_metric_memories(user_id: str, action: str, count: int = 1):
    """Emit memory processing metric."""
    if METRICS_ENABLED:
        consolidation_memories_processed.labels(user_id=user_id, action=action).inc(count)


@celery_app.task(
    name="hippocampai.consolidation.run_daily_consolidation",
    bind=True,
    max_retries=3,
)
def run_daily_consolidation(self):
    """
    Main task: Run nightly memory consolidation for all active users.

    This is triggered by Celery Beat at 3:00 AM daily.
    """
    if not ACTIVE_CONSOLIDATION_ENABLED:
        logger.info("Active consolidation is disabled (ACTIVE_CONSOLIDATION_ENABLED=false)")
        return {"status": "disabled", "message": "Consolidation feature is turned off"}

    logger.info("Starting daily memory consolidation run")
    start_time = time.time()

    try:
        # Get list of active users (users with recent memory activity)
        active_users = get_active_users(lookback_hours=CONSOLIDATION_LOOKBACK_HOURS)
        logger.info(f"Found {len(active_users)} active users to process")

        if not active_users:
            logger.info("No active users found, skipping consolidation")
            return {"status": "success", "users_processed": 0}

        # Process each user in parallel using Celery chord
        # This allows us to run consolidation for multiple users concurrently
        job = group(
            consolidate_user_memories.s(
                user_id=user_id,
                lookback_hours=CONSOLIDATION_LOOKBACK_HOURS,
                dry_run=CONSOLIDATION_DRY_RUN,
            )
            for user_id in active_users
        )

        # Execute and wait for all user consolidations to complete
        result = job.apply_async()

        # In production, you might not want to block here
        # Instead, return immediately and track progress separately
        results = result.get(timeout=600)  # 10 minute timeout

        # Aggregate results
        total_reviewed = sum(r.get("stats", {}).get("reviewed", 0) for r in results)
        total_deleted = sum(r.get("stats", {}).get("deleted", 0) for r in results)
        total_promoted = sum(r.get("stats", {}).get("promoted", 0) for r in results)

        duration = time.time() - start_time

        summary = {
            "status": "success",
            "users_processed": len(active_users),
            "total_memories_reviewed": total_reviewed,
            "total_memories_deleted": total_deleted,
            "total_memories_promoted": total_promoted,
            "duration_seconds": duration,
            "dry_run": CONSOLIDATION_DRY_RUN,
        }

        logger.info(f"Daily consolidation completed: {summary}")
        return summary

    except Exception as e:
        logger.exception(f"Daily consolidation failed: {e}")
        return {"status": "failed", "error": str(e)}


def cluster_memories(memories: list[Memory]) -> list[MemoryCluster]:
    """
    Group related memories into clusters for batch processing.

    Args:
        memories: List of memories to cluster

    Returns:
        List of MemoryCluster objects
    """
    clusters = []

    # Strategy 1: Cluster by session_id
    by_session: dict[str, list[Memory]] = {}
    no_session: list[Memory] = []

    for mem in memories:
        if mem.session_id:
            by_session.setdefault(mem.session_id, []).append(mem)
        else:
            no_session.append(mem)

    # Create clusters from sessions
    for session_id, session_memories in by_session.items():
        cluster = MemoryCluster(
            cluster_id=f"session-{session_id}",
            memories=[m.id for m in session_memories],
            theme=f"Session {session_id[:8]}",
            time_window_start=min(m.created_at for m in session_memories),
            time_window_end=max(m.created_at for m in session_memories),
            avg_importance=sum(m.importance for m in session_memories) / len(session_memories),
        )
        clusters.append(cluster)

    # Strategy 2: Cluster no-session memories by time windows (4-hour blocks)
    if no_session:
        # Sort by time
        no_session.sort(key=lambda m: m.created_at)

        # Group into 4-hour windows
        window_hours = 4
        current_cluster = []
        window_start = no_session[0].created_at

        for mem in no_session:
            if (mem.created_at - window_start).total_seconds() / 3600 <= window_hours:
                current_cluster.append(mem)
            else:
                # Start new cluster
                if current_cluster:
                    cluster = MemoryCluster(
                        cluster_id=f"time-{window_start.isoformat()}",
                        memories=[m.id for m in current_cluster],
                        theme=f"Memories from {window_start.strftime('%Y-%m-%d %H:%M')}",
                        time_window_start=current_cluster[0].created_at,
                        time_window_end=current_cluster[-1].created_at,
                        avg_importance=sum(m.importance for m in current_cluster) / len(current_cluster),
                    )
                    clusters.append(cluster)

                current_cluster = [mem]
                window_start = mem.created_at

        # Add final cluster
        if current_cluster:
            cluster = MemoryCluster(
                cluster_id=f"time-{window_start.isoformat()}",
                memories=[m.id for m in current_cluster],
                theme=f"Memories from {window_start.strftime('%Y-%m-%d %H:%M')}",
                time_window_start=current_cluster[0].created_at,
                time_window_end=current_cluster[-1].created_at,
                avg_importance=sum(m.importance for m in current_cluster) / len(current_cluster),
            )
            clusters.append(cluster)

    logger.info(f"Created {len(clusters)} clusters from {len(memories)} memories")
    return clusters


def _run_consolidation_sync(
    user_id: str,
    lookback_hours: int = 24,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Core consolidation logic (synchronous).

    Args:
        user_id: User ID to consolidate
        lookback_hours: How many hours back to review
        dry_run: If True, don't make changes

    Returns:
        Dictionary with consolidation results
    """
    logger.info(f"Starting consolidation for user {user_id} (lookback={lookback_hours}h, dry_run={dry_run})")
    start_time = time.time()

    # Create consolidation run record
    run = ConsolidationRun(
        user_id=user_id,
        lookback_hours=lookback_hours,
        dry_run=dry_run,
        status=ConsolidationStatus.RUNNING,
    )

    # Persist run to database
    from hippocampai.consolidation.db import get_db
    db = get_db()
    db.create_run(run)
    logger.debug(f"Created consolidation run {run.id} in database")

    try:
        # Step 1: Collect recent memories
        memories = collect_recent_memories(user_id, lookback_hours)
        logger.info(f"Collected {len(memories)} recent memories for {user_id}")

        if not memories:
            run.status = ConsolidationStatus.COMPLETED
            run.completed_at = datetime.now(timezone.utc)
            run.duration_seconds = time.time() - start_time
            logger.info(f"No memories to consolidate for {user_id}")
            emit_metric_run(user_id, "completed")

            # Update run in database
            db.update_run(
                run.id,
                {
                    "status": run.status.value,
                    "completed_at": run.completed_at,
                    "duration_seconds": run.duration_seconds,
                },
            )
            return run.dict()

        run.memories_reviewed = len(memories)

        # Step 2: Cluster memories
        clusters = cluster_memories(memories)
        run.clusters_created = len(clusters)
        logger.info(f"Created {len(clusters)} memory clusters for {user_id}")

        # Step 3: Process each cluster with LLM
        all_decisions = {"promoted_facts": [], "low_value_memory_ids": [], "updated_memories": [], "synthetic_memories": []}

        for cluster in clusters:
            try:
                memories_in_cluster = [m for m in memories if m.id in cluster.memories]

                # Call LLM for review
                llm_decision = llm_review_cluster(memories_in_cluster, user_id, cluster.theme)
                run.llm_calls_made += 1

                # Merge decisions
                for key in all_decisions.keys():
                    if key in llm_decision:
                        if isinstance(llm_decision[key], list):
                            all_decisions[key].extend(llm_decision[key])

            except Exception as e:
                logger.exception(f"Failed to process cluster {cluster.cluster_id} for {user_id}: {e}")
                continue

        # Step 4: Apply decisions with policy validation
        policy = ConsolidationPolicy()
        actions = apply_consolidation_decisions(
            memories=memories,
            llm_decisions=all_decisions,
            policy=policy,
            dry_run=dry_run,
        )

        # Step 5: Persist changes (if not dry run)
        if not dry_run:
            persist_consolidation_changes(user_id, actions)

        # Update run statistics
        run.memories_deleted = actions["stats"]["deleted"]
        run.memories_archived = actions["stats"]["archived"]
        run.memories_promoted = actions["stats"]["promoted"]
        run.memories_updated = actions["stats"]["updated"]
        run.memories_synthesized = actions["stats"]["synthesized"]

        # Emit metrics
        emit_metric_memories(user_id, "deleted", run.memories_deleted)
        emit_metric_memories(user_id, "promoted", run.memories_promoted)
        emit_metric_memories(user_id, "synthesized", run.memories_synthesized)

        # Complete run
        run.status = ConsolidationStatus.COMPLETED
        run.completed_at = datetime.now(timezone.utc)
        run.duration_seconds = time.time() - start_time

        # Generate dream report
        run.dream_report = generate_dream_report(run, actions)

        emit_metric_run(user_id, "completed")
        emit_metric_duration(user_id, run.duration_seconds)

        # Update run in database with final stats
        db.update_run(
            run.id,
            {
                "status": run.status.value,
                "completed_at": run.completed_at,
                "duration_seconds": run.duration_seconds,
                "memories_reviewed": run.memories_reviewed,
                "memories_deleted": run.memories_deleted,
                "memories_archived": run.memories_archived,
                "memories_promoted": run.memories_promoted,
                "memories_updated": run.memories_updated,
                "memories_synthesized": run.memories_synthesized,
                "clusters_created": run.clusters_created,
                "llm_calls_made": run.llm_calls_made,
                "dream_report": run.dream_report,
            },
        )

        logger.info(f"Consolidation completed for {user_id}: {run.dream_report}")
        return run.dict()

    except Exception as e:
        run.status = ConsolidationStatus.FAILED
        run.error_message = str(e)
        run.error_stacktrace = traceback.format_exc()
        run.completed_at = datetime.now(timezone.utc)
        run.duration_seconds = time.time() - start_time

        logger.exception(f"Consolidation failed for {user_id}: {e}")
        emit_metric_run(user_id, "failed")

        # Update run in database with error info
        db.update_run(
            run.id,
            {
                "status": run.status.value,
                "completed_at": run.completed_at,
                "duration_seconds": run.duration_seconds,
                "error_message": run.error_message,
                "error_stacktrace": run.error_stacktrace,
                "memories_reviewed": run.memories_reviewed,
                "clusters_created": run.clusters_created,
                "llm_calls_made": run.llm_calls_made,
            },
        )

        return run.dict()


@celery_app.task(
    name="hippocampai.consolidation.consolidate_user_memories",
    bind=True,
    max_retries=2,
)
def consolidate_user_memories(
    self,
    user_id: str,
    lookback_hours: int = 24,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Celery task wrapper for memory consolidation.

    Args:
        user_id: User ID to consolidate
        lookback_hours: How many hours back to review
        dry_run: If True, don't make changes

    Returns:
        Dictionary with consolidation results
    """
    return _run_consolidation_sync(user_id, lookback_hours, dry_run)


def get_active_users(lookback_hours: int = 24) -> list[str]:
    """
    Get list of users with recent memory activity.

    Args:
        lookback_hours: Hours to look back for activity

    Returns:
        List of user IDs with recent memory activity
    """
    try:
        from hippocampai.vector.qdrant_store import QdrantStore

        # Initialize Qdrant connection
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        store = QdrantStore(url=qdrant_url)

        # Calculate time threshold
        threshold = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)

        # Collect user IDs with recent activity
        active_user_ids = set()

        # Query both collections for recent memories
        collections = [store.collection_facts, store.collection_prefs]

        for collection_name in collections:
            try:
                # Scroll through all memories in batches
                # Note: Qdrant scroll doesn't support date range filters directly,
                # so we fetch memories and filter client-side
                batch_size = 1000
                offset = None

                while True:
                    # Use client.scroll directly for pagination support
                    results, next_offset = store.client.scroll(
                        collection_name=collection_name,
                        limit=batch_size,
                        offset=offset,
                        with_payload=True,
                        with_vectors=False,
                    )

                    if not results:
                        break

                    # Process this batch
                    for point in results:
                        payload = point.payload
                        user_id = payload.get("user_id")

                        if not user_id:
                            continue

                        # Check if memory was created or updated recently
                        created_at_str = payload.get("created_at")
                        updated_at_str = payload.get("updated_at")

                        # Parse dates
                        is_recent = False
                        if created_at_str:
                            try:
                                from dateutil import parser
                                created_at = parser.parse(created_at_str)
                                if created_at >= threshold:
                                    is_recent = True
                            except (ValueError, TypeError):
                                pass

                        if not is_recent and updated_at_str:
                            try:
                                from dateutil import parser
                                updated_at = parser.parse(updated_at_str)
                                if updated_at >= threshold:
                                    is_recent = True
                            except (ValueError, TypeError):
                                pass

                        if is_recent:
                            active_user_ids.add(user_id)

                    # Check if we have more results
                    if next_offset is None:
                        break
                    offset = next_offset

            except Exception as e:
                logger.warning(f"Failed to scan collection {collection_name}: {e}")
                continue

        active_users = list(active_user_ids)
        logger.info(
            f"Found {len(active_users)} active users with memory activity in last {lookback_hours}h"
        )
        return active_users

    except Exception as e:
        logger.exception(f"Failed to get active users: {e}")
        # Return empty list on failure (graceful degradation)
        return []


def collect_recent_memories(user_id: str, lookback_hours: int) -> list[Memory]:
    """
    Collect memories created/updated in the last N hours.

    Args:
        user_id: User ID
        lookback_hours: Hours to look back

    Returns:
        List of Memory objects
    """
    try:
        # Use MemoryClient to fetch memories
        from hippocampai.client import MemoryClient
        client = MemoryClient()

        # Calculate time threshold
        threshold = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)

        # Fetch all memories for user
        all_memories = client.get_memories(user_id=user_id, limit=1000)

        # Filter by date (client-side)
        recent_memories = []
        for mem in all_memories:
            try:
                # Check created_at
                if mem.created_at and mem.created_at >= threshold:
                    recent_memories.append(mem)
                    continue
                # Check updated_at
                if mem.updated_at and mem.updated_at >= threshold:
                    recent_memories.append(mem)
            except (TypeError, AttributeError):
                # If date comparison fails, include the memory
                recent_memories.append(mem)

        # Optionally filter by type (focus on transient memories)
        # recent_memories = [m for m in recent_memories if m.type in {MemoryType.EVENT, MemoryType.CONTEXT}]

        logger.info(f"Collected {len(recent_memories)} memories for {user_id} from last {lookback_hours}h")
        return recent_memories

    except Exception as e:
        logger.exception(f"Failed to collect memories for {user_id}: {e}")
        return []




def llm_review_cluster(
    cluster_memories: list[Memory],
    user_id: str,
    cluster_theme: str | None = None,
) -> dict[str, Any]:
    """
    Use LLM to review a cluster of memories and make consolidation decisions.

    Args:
        cluster_memories: Memories in this cluster
        user_id: User ID (for context)
        cluster_theme: Optional theme/topic

    Returns:
        ConsolidationDecision as dict
    """
    try:
        # Get dynamic user context based on cluster being consolidated
        user_context = get_user_context(
            user_id=user_id,
            cluster_memories=cluster_memories,
            cluster_theme=cluster_theme,
        )

        # Build prompt
        prompt = build_consolidation_prompt(
            memories=cluster_memories,
            user_context=user_context,
            cluster_theme=cluster_theme,
        )

        # Call LLM for consolidation decision
        response_text = call_llm_for_consolidation(
            prompt=prompt,
            model=CONSOLIDATION_LLM_MODEL,
            temperature=CONSOLIDATION_LLM_TEMPERATURE,
        )

        # Parse JSON response
        try:
            decision = json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {response_text[:200]}")
            raise ValueError(f"LLM returned invalid JSON: {e}")

        logger.debug(f"LLM decision for cluster: {len(decision.get('promoted_facts', []))} promoted, {len(decision.get('low_value_memory_ids', []))} to archive")

        return decision

    except Exception as e:
        logger.exception(f"LLM review failed for cluster: {e}")
        # Return empty decision on error
        return {
            "promoted_facts": [],
            "low_value_memory_ids": [],
            "updated_memories": [],
            "synthetic_memories": [],
            "reasoning": f"Error: {str(e)}",
        }


def call_llm_for_consolidation(prompt: str, model: str, temperature: float) -> str:
    """
    Call LLM API for consolidation using configured provider.

    Supports OpenAI, Groq, Anthropic, and Ollama providers via environment variables.

    Args:
        prompt: The consolidation prompt
        model: Model name (overrides env default)
        temperature: Sampling temperature

    Returns:
        LLM response text (JSON string)

    Environment Variables:
        - OPENAI_API_KEY: OpenAI API key
        - GROQ_API_KEY: Groq API key
        - ANTHROPIC_API_KEY: Anthropic API key
        - CONSOLIDATION_LLM_PROVIDER: Provider name (openai, groq, anthropic, ollama)
    """
    from hippocampai.consolidation.prompts import CONSOLIDATION_SYSTEM_MESSAGE

    try:
        # Determine LLM provider from env or model name
        provider = os.getenv("CONSOLIDATION_LLM_PROVIDER", "openai").lower()

        # Auto-detect provider from model name if not explicitly set
        if "gpt" in model.lower():
            provider = "openai"
        elif "llama" in model.lower() or "mixtral" in model.lower():
            provider = "groq"
        elif "claude" in model.lower():
            provider = "anthropic"

        logger.info(f"Using LLM provider: {provider}, model: {model}")

        # Initialize LLM client based on provider
        llm_client = None

        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set in environment")
            from hippocampai.adapters.provider_openai import OpenAILLM

            llm_client = OpenAILLM(api_key=api_key, model=model)

        elif provider == "groq":
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY not set in environment")
            from hippocampai.adapters.provider_groq import GroqLLM

            llm_client = GroqLLM(api_key=api_key, model=model)

        elif provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not set in environment")
            from hippocampai.adapters.provider_anthropic import AnthropicLLM

            llm_client = AnthropicLLM(api_key=api_key, model=model)

        elif provider == "ollama":
            from hippocampai.adapters.provider_ollama import OllamaLLM

            ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
            llm_client = OllamaLLM(base_url=ollama_url, model=model)

        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

        if not llm_client:
            raise ValueError(f"Failed to initialize LLM client for provider: {provider}")

        # Call LLM with consolidation prompt
        # Use generate() method with system message
        max_tokens = 4096  # Large enough for complex consolidation decisions
        response_text = llm_client.generate(
            prompt=prompt,
            system=CONSOLIDATION_SYSTEM_MESSAGE,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        if not response_text:
            logger.error("LLM returned empty response")
            return json.dumps(
                {
                    "promoted_facts": [],
                    "low_value_memory_ids": [],
                    "updated_memories": [],
                    "synthetic_memories": [],
                    "reasoning": "LLM returned empty response",
                }
            )

        logger.debug(f"LLM response length: {len(response_text)} chars")
        return response_text

    except Exception as e:
        logger.exception(f"Failed to call LLM for consolidation: {e}")
        # Return empty decision on error (graceful degradation)
        return json.dumps(
            {
                "promoted_facts": [],
                "low_value_memory_ids": [],
                "updated_memories": [],
                "synthetic_memories": [],
                "reasoning": f"Error calling LLM: {str(e)}",
            }
        )


def get_user_context(
    user_id: str,
    cluster_memories: list[Memory] | None = None,
    cluster_theme: str | None = None,
) -> dict[str, Any]:
    """
    Get dynamic user context for LLM prompt based on memories being consolidated.

    This function retrieves relevant user context by analyzing the cluster memories
    and performing semantic searches for related preferences, goals, and facts.

    Args:
        user_id: User ID
        cluster_memories: Memories being consolidated (for dynamic context)
        cluster_theme: Theme of the cluster (for focused retrieval)

    Returns:
        Dictionary with user context
    """
    try:
        from hippocampai.client import MemoryClient
        from hippocampai.models.memory import MemoryType

        client = MemoryClient()

        # Extract dynamic topics from cluster memories
        cluster_topics = []
        if cluster_memories:
            # Combine memory texts to understand what's being discussed
            cluster_texts = " ".join([mem.text for mem in cluster_memories[:5]])  # Sample first 5
            cluster_topics.append(cluster_texts[:200])  # Use first 200 chars as topic

        if cluster_theme:
            cluster_topics.append(cluster_theme)

        # Build dynamic semantic query based on cluster content
        semantic_query = " ".join(cluster_topics) if cluster_topics else "user information"

        logger.debug(f"Dynamic context query for {user_id}: '{semantic_query[:100]}...'")

        # Fetch semantically relevant preferences
        preferences = client.recall_memories(
            query=semantic_query,
            user_id=user_id,
            k=15,
        )
        preference_texts = [
            mem.memory.text for mem in preferences
            if mem.memory.type == MemoryType.PREFERENCE
        ][:5]  # Top 5 most relevant

        # Fetch semantically relevant goals
        goals = client.recall_memories(
            query=semantic_query,
            user_id=user_id,
            k=15,
        )
        goal_texts = [
            mem.memory.text for mem in goals
            if mem.memory.type == MemoryType.GOAL
        ][:5]  # Top 5 most relevant

        # Fetch semantically relevant facts
        facts = client.recall_memories(
            query=semantic_query,
            user_id=user_id,
            k=15,
        )
        fact_texts = [
            mem.memory.text for mem in facts
            if mem.memory.type == MemoryType.FACT
        ][:5]  # Top 5 most relevant

        # Fetch semantically relevant habits
        habits = client.recall_memories(
            query=semantic_query,
            user_id=user_id,
            k=10,
        )
        habit_texts = [
            mem.memory.text for mem in habits
            if mem.memory.type == MemoryType.HABIT
        ][:3]  # Top 3 most relevant

        # Get recent high-importance memories (temporal context)
        # Use broader query for recency check
        recent_important = client.recall_memories(
            query=semantic_query,
            user_id=user_id,
            k=30,
        )

        # Filter for recent AND important, excluding cluster memories
        cluster_ids = {mem.id for mem in cluster_memories} if cluster_memories else set()
        recent_topics = [
            mem.memory.text for mem in recent_important
            if mem.memory.importance >= 7.0
            and mem.memory.id not in cluster_ids
            and (datetime.now(timezone.utc) - mem.memory.created_at).days <= 7
        ][:5]  # Top 5 recent important memories from last 7 days

        # Calculate time window of cluster
        time_window = None
        if cluster_memories:
            timestamps = [mem.created_at for mem in cluster_memories]
            time_window = {
                "start": min(timestamps).isoformat(),
                "end": max(timestamps).isoformat(),
                "span_hours": (max(timestamps) - min(timestamps)).total_seconds() / 3600,
            }

        context = {
            "user_id": user_id,
            "preferences": preference_texts,
            "goals": goal_texts,
            "facts": fact_texts,
            "habits": habit_texts,
            "recent_topics": recent_topics,
            "cluster_theme": cluster_theme,
            "time_window": time_window,
            "has_preferences": len(preference_texts) > 0,
            "has_goals": len(goal_texts) > 0,
            "has_facts": len(fact_texts) > 0,
            "context_retrieval": "dynamic_semantic",  # Indicate dynamic retrieval
        }

        logger.info(
            f"Retrieved dynamic context for {user_id} (theme: {cluster_theme}): "
            f"{len(preference_texts)} preferences, "
            f"{len(goal_texts)} goals, "
            f"{len(fact_texts)} facts, "
            f"{len(habit_texts)} habits, "
            f"{len(recent_topics)} recent topics"
        )

        return context

    except Exception as e:
        logger.exception(f"Failed to retrieve user context for {user_id}: {e}")
        # Return minimal context on error
        return {
            "user_id": user_id,
            "preferences": [],
            "goals": [],
            "facts": [],
            "habits": [],
            "recent_topics": [],
            "cluster_theme": cluster_theme,
            "has_preferences": False,
            "has_goals": False,
            "has_facts": False,
            "context_retrieval": "error_fallback",
        }


def persist_consolidation_changes(user_id: str, actions: dict[str, Any]) -> None:
    """
    Persist consolidation changes to database and vector store.

    Args:
        user_id: User ID
        actions: Actions dictionary from apply_consolidation_decisions
    """
    from hippocampai.client import MemoryClient
    client = MemoryClient()

    # Delete memories
    for delete_action in actions["to_delete"]:
        try:
            client.delete_memory(memory_id=delete_action["id"], user_id=user_id)
            logger.info(f"Deleted memory {delete_action['id']}: {delete_action['reason']}")
        except Exception as e:
            logger.exception(f"Failed to delete memory {delete_action['id']}: {e}")

    # Archive memories (soft delete by setting is_archived=True)
    for archive_action in actions["to_archive"]:
        try:
            # Update memory with is_archived flag
            # NOTE: You may need to implement this in your update_memory method
            client.update_memory(
                memory_id=archive_action["id"],
                updates={"is_archived": True, "archived_at": datetime.now(timezone.utc).isoformat()},
            )
            logger.info(f"Archived memory {archive_action['id']}: {archive_action['reason']}")
        except Exception as e:
            logger.exception(f"Failed to archive memory {archive_action['id']}: {e}")

    # Promote memories (update importance)
    for promote_action in actions["to_promote"]:
        try:
            client.update_memory(
                memory_id=promote_action["id"],
                updates={
                    "importance": promote_action["new_importance"],
                    "promotion_count": "+1",  # Increment
                },
            )
            logger.info(f"Promoted memory {promote_action['id']} to importance {promote_action['new_importance']}")
        except Exception as e:
            logger.exception(f"Failed to promote memory {promote_action['id']}: {e}")

    # Update memories
    for update_action in actions["to_update"]:
        try:
            updates = {}
            if "new_text" in update_action:
                updates["text"] = update_action["new_text"]
            if "new_importance" in update_action:
                updates["importance"] = update_action["new_importance"]

            client.update_memory(memory_id=update_action["id"], updates=updates)
            logger.info(f"Updated memory {update_action['id']}")
        except Exception as e:
            logger.exception(f"Failed to update memory {update_action['id']}: {e}")

    # Create synthetic memories
    for synthetic in actions["to_create"]:
        try:
            new_memory = client.create_memory(
                text=synthetic["text"],
                user_id=user_id,
                memory_type=synthetic.get("type", "context"),
                importance=synthetic.get("importance", 7.0),
                tags=synthetic.get("tags", []),
                metadata={
                    "source_memory_ids": synthetic.get("source_ids", []),
                    "is_synthetic": True,
                    "consolidation_generated": True,
                },
            )
            logger.info(f"Created synthetic memory {new_memory.id}: {synthetic['text'][:50]}...")
        except Exception as e:
            logger.exception(f"Failed to create synthetic memory: {e}")


def generate_dream_report(run: ConsolidationRun, actions: dict[str, Any]) -> str:
    """
    Generate a human-readable dream report.

    Args:
        run: ConsolidationRun object
        actions: Actions taken

    Returns:
        Dream report string
    """
    report = f"""Consolidation completed for user {run.user_id}:
- Reviewed: {run.memories_reviewed} memories
- Deleted: {run.memories_deleted}
- Archived: {run.memories_archived}
- Promoted: {run.memories_promoted}
- Updated: {run.memories_updated}
- Synthesized: {run.memories_synthesized} new memories
- Duration: {run.duration_seconds:.2f}s
- Clusters: {run.clusters_created}
- LLM calls: {run.llm_calls_made}
"""

    if run.dry_run:
        report += "\n[DRY RUN - No changes were made]"

    return report.strip()
