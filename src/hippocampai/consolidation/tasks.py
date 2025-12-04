"""Celery tasks for memory consolidation (Sleep Phase)."""

import json
import logging
import os
import time
import traceback
from datetime import datetime, timedelta, timezone
from typing import Any

from celery import chain, chord, group
from hippocampai.celery_app import celery_app
from hippocampai.client import MemoryClient
from hippocampai.consolidation.models import (
    ConsolidationDecision,
    ConsolidationRun,
    ConsolidationStatus,
    MemoryCluster,
)
from hippocampai.consolidation.policy import (
    ConsolidationPolicy,
    ConsolidationPolicyEngine,
    apply_consolidation_decisions,
)
from hippocampai.consolidation.prompts import (
    CONSOLIDATION_SYSTEM_MESSAGE,
    build_consolidation_prompt,
    build_cluster_theme_prompt,
)
from hippocampai.models import Memory, MemoryType

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
    Consolidate memories for a single user.

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
                cluster_memories = [m for m in memories if m.id in cluster.memories]

                # Call LLM for review
                llm_decision = llm_review_cluster(cluster_memories, user_id, cluster.theme)
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

        return run.dict()


def get_active_users(lookback_hours: int = 24) -> list[str]:
    """
    Get list of users with recent memory activity.

    Args:
        lookback_hours: Hours to look back for activity

    Returns:
        List of user IDs
    """
    # This would query your database for users with recent activity
    # For now, we'll use a placeholder implementation

    # TODO: Implement actual DB query
    # Example SQL:
    # SELECT DISTINCT user_id FROM memories
    # WHERE created_at > NOW() - INTERVAL '{lookback_hours} hours'
    #    OR updated_at > NOW() - INTERVAL '{lookback_hours} hours'

    logger.warning("get_active_users: Using placeholder implementation")
    return []  # Replace with actual user IDs


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
        # Use HippocampAI client to fetch memories
        client = HippocampAIClient()

        # Calculate time threshold
        threshold = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)

        # Fetch recent memories
        # NOTE: This assumes you have a method to filter by date
        # You may need to implement this in your client/backend
        all_memories = client.recall_memories(
            query="",  # Empty query to get all
            user_id=user_id,
            k=1000,  # Generous limit
        )

        # Filter by date (client-side for now)
        recent_memories = [
            mem.memory
            for mem in all_memories
            if mem.memory.created_at >= threshold or mem.memory.updated_at >= threshold
        ]

        # Optionally filter by type (focus on transient memories)
        # recent_memories = [m for m in recent_memories if m.type in {MemoryType.EVENT, MemoryType.CONTEXT}]

        logger.info(f"Collected {len(recent_memories)} memories for {user_id} from last {lookback_hours}h")
        return recent_memories

    except Exception as e:
        logger.exception(f"Failed to collect memories for {user_id}: {e}")
        return []


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
        # Get user context (preferences, goals, etc.)
        user_context = get_user_context(user_id)

        # Build prompt
        prompt = build_consolidation_prompt(
            memories=cluster_memories,
            user_context=user_context,
            cluster_theme=cluster_theme,
        )

        # Call LLM via your UnifiedClient or similar
        # This is a placeholder - replace with your actual LLM client
        client = HippocampAIClient()

        # Use the chat/completion endpoint
        # NOTE: Adjust based on your actual LLM client interface
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
    Call LLM API for consolidation.

    This is a placeholder - replace with your actual LLM client.

    Args:
        prompt: The consolidation prompt
        model: Model name
        temperature: Sampling temperature

    Returns:
        LLM response text (JSON string)
    """
    # TODO: Implement actual LLM call using your UnifiedClient or OpenAI/Anthropic client
    # Example using OpenAI:
    #
    # import openai
    # response = openai.ChatCompletion.create(
    #     model=model,
    #     messages=[
    #         {"role": "system", "content": CONSOLIDATION_SYSTEM_MESSAGE},
    #         {"role": "user", "content": prompt}
    #     ],
    #     temperature=temperature,
    #     response_format={"type": "json_object"}  # For GPT-4 Turbo JSON mode
    # )
    # return response.choices[0].message.content

    logger.warning("call_llm_for_consolidation: Using placeholder implementation")
    # Return empty decision for now
    return json.dumps(
        {
            "promoted_facts": [],
            "low_value_memory_ids": [],
            "updated_memories": [],
            "synthetic_memories": [],
            "reasoning": "Placeholder LLM response",
        }
    )


def get_user_context(user_id: str) -> dict[str, Any]:
    """
    Get user context for LLM prompt (preferences, goals, etc.).

    Args:
        user_id: User ID

    Returns:
        Dictionary with user context
    """
    # TODO: Implement actual user context retrieval
    # This might include:
    # - User preferences
    # - Active goals
    # - Recent topics of interest
    # - User profile information

    return {
        "user_id": user_id,
        # Add more context as needed
    }


def persist_consolidation_changes(user_id: str, actions: dict[str, Any]) -> None:
    """
    Persist consolidation changes to database and vector store.

    Args:
        user_id: User ID
        actions: Actions dictionary from apply_consolidation_decisions
    """
    client = HippocampAIClient()

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
