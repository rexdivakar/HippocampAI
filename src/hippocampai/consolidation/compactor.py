"""Conversation Compactor - Consolidate and compact conversation memories."""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


@dataclass
class CompactionMetrics:
    """Metrics from a compaction operation."""

    # Input metrics
    input_memories: int = 0
    input_tokens: int = 0
    input_characters: int = 0

    # Output metrics
    output_memories: int = 0
    output_tokens: int = 0
    output_characters: int = 0

    # Compression stats
    compression_ratio: float = 0.0
    tokens_saved: int = 0
    memories_merged: int = 0

    # Processing stats
    clusters_found: int = 0
    llm_calls: int = 0
    duration_seconds: float = 0.0

    # Cost estimation (approximate)
    estimated_input_cost: float = 0.0
    estimated_output_cost: float = 0.0

    # Storage metrics
    estimated_storage_saved_bytes: int = 0
    avg_memory_size_before: float = 0.0
    avg_memory_size_after: float = 0.0

    # Type breakdown
    types_compacted: dict = field(default_factory=dict)

    # Quality metrics
    key_facts_preserved: int = 0
    entities_preserved: int = 0
    context_retention_score: float = 0.0


@dataclass
class CompactionResult:
    """Result of a compaction operation."""

    id: str = field(default_factory=lambda: str(uuid4()))
    user_id: str = ""
    session_id: Optional[str] = None
    status: str = "pending"

    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None

    metrics: CompactionMetrics = field(default_factory=CompactionMetrics)

    # Detailed actions
    actions: list[dict] = field(default_factory=list)

    # Summary
    summary: Optional[str] = None

    # Insights about the compaction
    insights: list[str] = field(default_factory=list)

    # Preserved key information
    preserved_facts: list[str] = field(default_factory=list)
    preserved_entities: list[str] = field(default_factory=list)

    # Configuration used
    config: dict = field(default_factory=dict)

    dry_run: bool = False
    error: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "status": self.status,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "metrics": {
                "input_memories": self.metrics.input_memories,
                "input_tokens": self.metrics.input_tokens,
                "input_characters": self.metrics.input_characters,
                "output_memories": self.metrics.output_memories,
                "output_tokens": self.metrics.output_tokens,
                "output_characters": self.metrics.output_characters,
                "compression_ratio": self.metrics.compression_ratio,
                "tokens_saved": self.metrics.tokens_saved,
                "memories_merged": self.metrics.memories_merged,
                "clusters_found": self.metrics.clusters_found,
                "llm_calls": self.metrics.llm_calls,
                "duration_seconds": self.metrics.duration_seconds,
                "estimated_input_cost": self.metrics.estimated_input_cost,
                "estimated_output_cost": self.metrics.estimated_output_cost,
                "estimated_storage_saved_bytes": self.metrics.estimated_storage_saved_bytes,
                "avg_memory_size_before": self.metrics.avg_memory_size_before,
                "avg_memory_size_after": self.metrics.avg_memory_size_after,
                "types_compacted": self.metrics.types_compacted,
                "key_facts_preserved": self.metrics.key_facts_preserved,
                "entities_preserved": self.metrics.entities_preserved,
                "context_retention_score": self.metrics.context_retention_score,
            },
            "actions": self.actions,
            "summary": self.summary,
            "insights": self.insights,
            "preserved_facts": self.preserved_facts,
            "preserved_entities": self.preserved_entities,
            "config": self.config,
            "dry_run": self.dry_run,
            "error": self.error,
        }


def estimate_tokens(text: str) -> int:
    """Estimate token count (roughly 4 chars per token for English)."""
    return len(text) // 4


def estimate_cost(tokens: int, is_input: bool = True) -> float:
    """Estimate cost in USD (using GPT-4 pricing as reference)."""
    # GPT-4 pricing: $0.03/1K input, $0.06/1K output
    rate = 0.00003 if is_input else 0.00006
    return tokens * rate


class ConversationCompactor:
    """Compact and consolidate conversation memories."""

    # Memory types that can be compacted
    COMPACTABLE_TYPES = ["fact", "event", "context", "preference", "goal", "habit"]

    def __init__(
        self,
        qdrant_url: str | None = None,
        llm_provider: str = "groq",
    ):
        """Initialize the compactor.

        Args:
            qdrant_url: Qdrant server URL
            llm_provider: LLM provider for summarization (groq, openai, ollama)
        """
        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self.llm_provider = llm_provider

        from qdrant_client import QdrantClient

        self.client = QdrantClient(url=self.qdrant_url)

    def compact_conversations(
        self,
        user_id: str,
        session_id: Optional[str] = None,
        lookback_hours: int = 168,  # 1 week
        min_memories: int = 5,
        dry_run: bool = False,
        memory_types: Optional[list[str]] = None,  # Filter by types
    ) -> CompactionResult:
        """Compact conversation memories into summaries.

        Args:
            user_id: User ID to compact memories for
            session_id: Optional session ID filter
            lookback_hours: How far back to look for memories
            min_memories: Minimum memories needed to trigger compaction
            dry_run: If True, don't make changes
            memory_types: Optional list of memory types to compact (e.g., ["fact", "event"])

        Returns:
            CompactionResult with metrics and actions
        """
        import time

        result = CompactionResult(
            user_id=user_id,
            session_id=session_id,
            dry_run=dry_run,
            config={
                "lookback_hours": lookback_hours,
                "min_memories": min_memories,
                "memory_types": memory_types or self.COMPACTABLE_TYPES,
            },
        )

        start_time = time.time()

        try:
            # 1. Collect memories (with type filtering)
            memories = self._collect_memories(user_id, session_id, lookback_hours, memory_types)

            result.metrics.input_memories = len(memories)
            result.metrics.input_characters = sum(len(m.get("text", "")) for m in memories)
            result.metrics.input_tokens = sum(estimate_tokens(m.get("text", "")) for m in memories)
            result.metrics.estimated_input_cost = estimate_cost(
                result.metrics.input_tokens, is_input=True
            )

            # Calculate average memory size before
            if memories:
                result.metrics.avg_memory_size_before = result.metrics.input_characters / len(
                    memories
                )

            # Count types
            type_counts: dict[str, int] = {}
            for m in memories:
                t = m.get("type", "unknown")
                type_counts[t] = type_counts.get(t, 0) + 1
            result.metrics.types_compacted = type_counts

            if len(memories) < min_memories:
                result.status = "skipped"
                result.summary = (
                    f"Not enough memories to compact ({len(memories)} < {min_memories})"
                )
                result.insights.append(
                    f"Found {len(memories)} memories, need at least {min_memories} to compact"
                )
                result.completed_at = datetime.now(timezone.utc)
                result.metrics.duration_seconds = time.time() - start_time
                return result

            # 2. Cluster by topic/time
            clusters = self._cluster_memories(memories)
            result.metrics.clusters_found = len(clusters)

            # 3. Summarize each cluster and extract key facts
            summaries = []
            all_key_facts = []
            all_entities = []

            for cluster in clusters:
                if len(cluster) >= 2:
                    summary = self._summarize_cluster(cluster, dry_run)
                    if summary:
                        summaries.append(summary)
                        result.metrics.llm_calls += 1

                        # Extract key facts from metadata
                        key_facts = summary.get("metadata", {}).get("key_facts", [])
                        all_key_facts.extend(key_facts)

                        # Extract entities
                        entities = self._extract_entities_from_cluster(cluster)
                        all_entities.extend(entities)

                        result.actions.append(
                            {
                                "action": "summarize",
                                "input_count": len(cluster),
                                "input_tokens": sum(
                                    estimate_tokens(m.get("text", "")) for m in cluster
                                ),
                                "output_tokens": estimate_tokens(summary.get("text", "")),
                                "preview": summary.get("text", "")[:200],
                                "key_facts_found": len(key_facts),
                            }
                        )

            # Store preserved facts and entities
            result.preserved_facts = list(set(all_key_facts))[:20]  # Limit to 20
            result.preserved_entities = list(set(all_entities))[:20]
            result.metrics.key_facts_preserved = len(result.preserved_facts)
            result.metrics.entities_preserved = len(result.preserved_entities)

            # 4. Calculate output metrics
            result.metrics.output_memories = len(summaries)
            result.metrics.output_characters = sum(len(s.get("text", "")) for s in summaries)
            result.metrics.output_tokens = sum(
                estimate_tokens(s.get("text", "")) for s in summaries
            )
            result.metrics.estimated_output_cost = estimate_cost(
                result.metrics.output_tokens, is_input=False
            )

            # Calculate average memory size after
            if summaries:
                result.metrics.avg_memory_size_after = result.metrics.output_characters / len(
                    summaries
                )

            # 5. Calculate compression stats
            if result.metrics.input_tokens > 0:
                result.metrics.compression_ratio = 1 - (
                    result.metrics.output_tokens / result.metrics.input_tokens
                )
            result.metrics.tokens_saved = result.metrics.input_tokens - result.metrics.output_tokens
            result.metrics.memories_merged = (
                result.metrics.input_memories - result.metrics.output_memories
            )

            # Estimate storage saved (rough: 1 token ≈ 4 bytes + overhead)
            result.metrics.estimated_storage_saved_bytes = result.metrics.tokens_saved * 6

            # Calculate context retention score (how much key info was preserved)
            if result.metrics.input_memories > 0:
                result.metrics.context_retention_score = min(
                    1.0,
                    (result.metrics.key_facts_preserved + result.metrics.entities_preserved)
                    / (result.metrics.input_memories * 0.5),
                )

            # 6. Generate insights
            result.insights = self._generate_insights(result, memories, summaries)

            # 7. Store summaries (if not dry run)
            if not dry_run and summaries:
                self._store_summaries(summaries, user_id, session_id)
                self._archive_original_memories(memories)
                result.actions.append(
                    {
                        "action": "store_summaries",
                        "count": len(summaries),
                    }
                )
                result.actions.append(
                    {
                        "action": "archive_originals",
                        "count": len(memories),
                    }
                )

            result.status = "completed"
            result.summary = self._generate_summary(result)

        except Exception as e:
            logger.exception(f"Compaction failed: {e}")
            result.status = "failed"
            result.error = str(e)

        result.completed_at = datetime.now(timezone.utc)
        result.metrics.duration_seconds = time.time() - start_time

        return result

    def _extract_entities_from_cluster(self, cluster: list[dict]) -> list[str]:
        """Extract entity names from a cluster of memories."""
        entities = []
        for m in cluster:
            # Simple entity extraction - look for capitalized words
            words = m.get("text", "").split()
            for word in words:
                if word and word[0].isupper() and len(word) > 2 and word.isalpha():
                    entities.append(word)
            # Also check metadata for entities
            if m.get("metadata", {}).get("entities"):
                entities.extend(m["metadata"]["entities"])
        return entities

    def _generate_insights(
        self, result: CompactionResult, memories: list[dict], summaries: list[dict]
    ) -> list[str]:
        """Generate insights about the compaction."""
        insights = []
        m = result.metrics

        # Compression insight
        if m.compression_ratio > 0.8:
            insights.append(
                f"🎯 Excellent compression! Reduced token usage by {m.compression_ratio * 100:.0f}%"
            )
        elif m.compression_ratio > 0.5:
            insights.append(
                f"✅ Good compression achieved: {m.compression_ratio * 100:.0f}% reduction"
            )
        elif m.compression_ratio > 0:
            insights.append(f"📊 Moderate compression: {m.compression_ratio * 100:.0f}% reduction")

        # Token savings insight
        if m.tokens_saved > 1000:
            insights.append(f"💰 Significant token savings: {m.tokens_saved:,} tokens saved")
        elif m.tokens_saved > 100:
            insights.append(f"💵 Token savings: {m.tokens_saved:,} tokens saved")

        # Memory consolidation insight
        if m.memories_merged > 10:
            insights.append(
                f"📦 Consolidated {m.memories_merged} memories into {m.output_memories} summaries"
            )

        # Cost insight
        total_cost = m.estimated_input_cost + m.estimated_output_cost
        if total_cost > 0:
            insights.append(f"💲 Estimated processing cost: ${total_cost:.4f}")

        # Context retention insight
        if m.context_retention_score > 0.8:
            insights.append(
                f"🧠 High context retention: {m.context_retention_score * 100:.0f}% of key information preserved"
            )
        elif m.context_retention_score > 0.5:
            insights.append(
                f"📝 Good context retention: {m.context_retention_score * 100:.0f}% of key information preserved"
            )

        # Type breakdown insight
        if m.types_compacted:
            type_str = ", ".join([f"{k}: {v}" for k, v in m.types_compacted.items()])
            insights.append(f"📋 Memory types compacted: {type_str}")

        # Future benefit insight
        future_savings = m.tokens_saved * 10  # Assume 10 future retrievals
        if future_savings > 0:
            insights.append(
                f"🚀 Projected savings: ~{future_savings:,} tokens over next 10 retrievals"
            )

        return insights

    def _collect_memories(
        self,
        user_id: str,
        session_id: Optional[str],
        lookback_hours: int,
        memory_types: Optional[list[str]] = None,
    ) -> list[dict]:
        """Collect memories for compaction.

        Args:
            user_id: User ID
            session_id: Optional session filter
            lookback_hours: How far back to look
            memory_types: Optional list of types to include
        """
        from datetime import timedelta

        from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

        cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
        allowed_types = memory_types or self.COMPACTABLE_TYPES

        collections = ["hippocampai_facts", "hippocampai_prefs"]
        all_memories = []

        for collection in collections:
            try:
                if not self.client.collection_exists(collection):
                    continue

                # Build filter
                conditions = [FieldCondition(key="user_id", match=MatchValue(value=user_id))]

                if session_id:
                    conditions.append(
                        FieldCondition(key="session_id", match=MatchValue(value=session_id))
                    )

                # Add type filter if specified
                if memory_types:
                    conditions.append(FieldCondition(key="type", match=MatchAny(any=memory_types)))

                from typing import cast as _cast

                from qdrant_client.models import Condition

                results, _ = self.client.scroll(
                    collection_name=collection,
                    scroll_filter=Filter(must=_cast(list[Condition], conditions)),
                    limit=1000,
                    with_payload=True,
                )

                for r in results:
                    payload = r.payload
                    if payload is None:
                        continue

                    # Skip archived memories
                    if payload.get("is_archived"):
                        continue

                    # Filter by type (double-check)
                    mem_type = payload.get("type", "fact")
                    if mem_type not in allowed_types:
                        continue

                    # Filter by time
                    created_at = payload.get("created_at")
                    if created_at:
                        if isinstance(created_at, str):
                            try:
                                created_dt = datetime.fromisoformat(
                                    created_at.replace("Z", "+00:00")
                                )
                                if created_dt < cutoff:
                                    continue
                            except (ValueError, TypeError):
                                # Malformed date, include memory anyway
                                pass

                    all_memories.append({"id": str(r.id), "collection": collection, **payload})

            except Exception as e:
                logger.warning(f"Error collecting from {collection}: {e}")

        # Sort by created_at
        all_memories.sort(key=lambda m: m.get("created_at", ""), reverse=False)

        return all_memories

    def _cluster_memories(self, memories: list[dict]) -> list[list[dict]]:
        """Cluster memories by topic and time proximity."""
        if not memories:
            return []

        # Simple clustering: group by session_id and time windows
        clusters = []
        current_cluster = []
        last_time = None

        for memory in memories:
            created_at = memory.get("created_at")
            if isinstance(created_at, str):
                try:
                    current_time = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                except Exception:
                    current_time = None
            else:
                current_time = created_at

            # Start new cluster if time gap > 1 hour or cluster too large
            if last_time and current_time:
                time_gap = (current_time - last_time).total_seconds() / 3600
                if time_gap > 1 or len(current_cluster) >= 10:
                    if current_cluster:
                        clusters.append(current_cluster)
                    current_cluster = []

            current_cluster.append(memory)
            last_time = current_time

        if current_cluster:
            clusters.append(current_cluster)

        return clusters

    def _summarize_cluster(self, cluster: list[dict], dry_run: bool) -> Optional[dict]:
        """Summarize a cluster of memories using LLM."""
        if not cluster:
            return None

        # Combine texts
        texts = []
        for m in cluster:
            text = m.get("text", "")
            if text:
                texts.append(text)

        if not texts:
            return None

        combined = "\n---\n".join(texts)

        # Generate summary using LLM
        try:
            summary_text = self._call_llm_for_summary(combined, dry_run)

            # Extract key facts
            key_facts = self._extract_key_facts(cluster)

            return {
                "text": summary_text,
                "type": "summary",
                "importance": max(m.get("importance", 5.0) for m in cluster),
                "tags": list(set(tag for m in cluster for tag in m.get("tags", []))),
                "metadata": {
                    "source": "compaction",
                    "original_count": len(cluster),
                    "original_ids": [m.get("id") for m in cluster],
                    "key_facts": key_facts,
                    "compacted_at": datetime.now(timezone.utc).isoformat(),
                },
            }
        except Exception as e:
            logger.warning(f"Failed to summarize cluster: {e}")
            return None

    def _call_llm_for_summary(self, text: str, dry_run: bool) -> str:
        """Call LLM to generate summary."""
        if dry_run:
            # Return a mock summary for dry run
            word_count = len(text.split())
            return f"[DRY RUN] Summary of {word_count} words of conversation content."

        prompt = f"""Summarize the following conversation memories into a concise summary.
Focus on:
- Key facts about the user (name, preferences, goals)
- Important decisions or events
- Recurring themes or topics

Keep the summary under 200 words.

Conversation:
{text[:4000]}

Summary:"""

        try:
            if self.llm_provider == "groq":
                return self._call_groq(prompt)
            elif self.llm_provider == "openai":
                return self._call_openai(prompt)
            else:
                return self._call_ollama(prompt)
        except Exception as e:
            logger.warning(f"LLM call failed: {e}")
            # Fallback: simple extraction
            return f"Conversation summary: {text[:500]}..."

    def _call_groq(self, prompt: str) -> str:
        """Call Groq API."""
        from groq import Groq

        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
        )
        content = response.choices[0].message.content
        return content if content is not None else ""

    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API."""
        from openai import OpenAI

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
        )
        content = response.choices[0].message.content
        return content if content is not None else ""

    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API."""
        import httpx

        response = httpx.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama2", "prompt": prompt, "stream": False},
            timeout=60,
        )
        result: str = response.json().get("response", "")
        return result

    def _extract_key_facts(self, cluster: list[dict]) -> list[str]:
        """Extract key facts from a cluster."""
        facts = []
        for m in cluster:
            text = m.get("text", "").lower()
            # Look for identity statements
            if "my name is" in text or "i am" in text or "i'm" in text:
                facts.append(f"Identity: {m.get('text', '')[:100]}")
            elif "i like" in text or "i love" in text or "i prefer" in text:
                facts.append(f"Preference: {m.get('text', '')[:100]}")
            elif "i work" in text or "my job" in text:
                facts.append(f"Work: {m.get('text', '')[:100]}")
        return facts[:5]  # Limit to 5 key facts

    def _store_summaries(
        self, summaries: list[dict], user_id: str, session_id: Optional[str]
    ) -> None:
        """Store summary memories in Qdrant."""
        from hippocampai.embed.embedder import get_embedder

        embedder = get_embedder()
        collection = "hippocampai_facts"

        for summary in summaries:
            try:
                memory_id = str(uuid4())
                vector = embedder.encode_single(summary["text"])

                payload = {
                    "id": memory_id,
                    "user_id": user_id,
                    "session_id": session_id,
                    "text": summary["text"],
                    "type": summary.get("type", "summary"),
                    "importance": summary.get("importance", 7.0),
                    "tags": summary.get("tags", ["summary", "compacted"]),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "metadata": summary.get("metadata", {}),
                }

                from qdrant_client.models import PointStruct

                self.client.upsert(
                    collection_name=collection,
                    points=[PointStruct(id=memory_id, vector=vector.tolist(), payload=payload)],
                )

            except Exception as e:
                logger.warning(f"Failed to store summary: {e}")

    def _archive_original_memories(self, memories: list[dict]) -> None:
        """Mark original memories as archived."""
        for memory in memories:
            try:
                collection = memory.get("collection", "hippocampai_facts")
                memory_id = memory.get("id")

                if memory_id:
                    self.client.set_payload(
                        collection_name=collection,
                        payload={
                            "is_archived": True,
                            "archived_at": datetime.now(timezone.utc).isoformat(),
                            "archived_reason": "compaction",
                        },
                        points=[memory_id],
                    )
            except Exception as e:
                logger.warning(f"Failed to archive memory {memory.get('id')}: {e}")

    def _generate_summary(self, result: CompactionResult) -> str:
        """Generate a human-readable summary."""
        m = result.metrics
        return f"""Compaction {"(DRY RUN) " if result.dry_run else ""}completed for user {result.user_id}:
- Input: {m.input_memories} memories ({m.input_tokens:,} tokens)
- Output: {m.output_memories} summaries ({m.output_tokens:,} tokens)
- Compression: {m.compression_ratio:.1%} ({m.tokens_saved:,} tokens saved)
- Clusters: {m.clusters_found}
- LLM calls: {m.llm_calls}
- Duration: {m.duration_seconds:.2f}s
- Est. cost: ${m.estimated_input_cost + m.estimated_output_cost:.4f}"""
