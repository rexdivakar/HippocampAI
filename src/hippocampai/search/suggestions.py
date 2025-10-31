"""Search suggestions based on user history."""

import logging
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Optional

from hippocampai.models.search import SearchSuggestion

logger = logging.getLogger(__name__)


class SearchSuggestionEngine:
    """Generate search suggestions based on user query history."""

    def __init__(self, min_frequency: int = 2, history_days: int = 90):
        """
        Initialize suggestion engine.

        Args:
            min_frequency: Minimum query frequency to suggest
            history_days: Days of history to consider
        """
        self.min_frequency = min_frequency
        self.history_days = history_days
        self._query_history: dict[str, list[tuple[str, datetime]]] = defaultdict(
            list
        )  # user_id -> [(query, timestamp)]
        self._query_frequencies: dict[str, Counter] = defaultdict(
            Counter
        )  # user_id -> Counter(query)

    def record_query(self, user_id: str, query: str, tags: Optional[list[str]] = None):
        """
        Record a search query.

        Args:
            user_id: User ID
            query: Search query
            tags: Optional tags associated with query
        """
        timestamp = datetime.now(timezone.utc)
        normalized_query = query.strip().lower()

        self._query_history[user_id].append((normalized_query, timestamp))
        self._query_frequencies[user_id][normalized_query] += 1

        logger.debug(f"Recorded query for user {user_id}: {normalized_query}")

    def get_suggestions(
        self, user_id: str, prefix: Optional[str] = None, limit: int = 5
    ) -> list[SearchSuggestion]:
        """
        Get search suggestions for a user.

        Args:
            user_id: User ID
            prefix: Optional prefix to filter suggestions (autocomplete)
            limit: Maximum number of suggestions

        Returns:
            List of SearchSuggestion objects sorted by relevance
        """
        # Clean old history
        self._clean_old_history(user_id)

        frequencies = self._query_frequencies.get(user_id, Counter())
        if not frequencies:
            return []

        # Filter by prefix if provided
        if prefix:
            prefix_lower = prefix.strip().lower()
            filtered = {q: freq for q, freq in frequencies.items() if q.startswith(prefix_lower)}
        else:
            filtered = dict(frequencies)

        # Filter by minimum frequency
        filtered = {q: freq for q, freq in filtered.items() if freq >= self.min_frequency}

        if not filtered:
            return []

        # Get last used timestamp for each query
        last_used_map: dict[str, datetime] = {}
        for query, timestamp in self._query_history[user_id]:
            if query in filtered:
                if query not in last_used_map or timestamp > last_used_map[query]:
                    last_used_map[query] = timestamp

        # Create suggestions
        suggestions = []
        max_freq = max(filtered.values())

        for query, freq in filtered.items():
            # Confidence based on frequency and recency
            freq_score = freq / max_freq if max_freq > 0 else 0.5
            last_used = last_used_map.get(query)

            # Recency boost (queries used recently get higher confidence)
            recency_score = 0.5
            if last_used:
                days_ago = (datetime.now(timezone.utc) - last_used).days
                recency_score = max(0.1, 1.0 - (days_ago / self.history_days))

            confidence = (freq_score * 0.6) + (recency_score * 0.4)

            suggestions.append(
                SearchSuggestion(
                    query=query, confidence=confidence, frequency=freq, last_used=last_used
                )
            )

        # Sort by confidence (highest first)
        suggestions.sort(key=lambda s: s.confidence, reverse=True)
        return suggestions[:limit]

    def get_popular_queries(self, user_id: str, limit: int = 10) -> list[SearchSuggestion]:
        """
        Get most popular queries for a user.

        Args:
            user_id: User ID
            limit: Maximum number of queries

        Returns:
            List of SearchSuggestion objects sorted by frequency
        """
        self._clean_old_history(user_id)

        frequencies = self._query_frequencies.get(user_id, Counter())
        if not frequencies:
            return []

        # Get last used timestamp
        last_used_map: dict[str, datetime] = {}
        for query, timestamp in self._query_history[user_id]:
            if query not in last_used_map or timestamp > last_used_map[query]:
                last_used_map[query] = timestamp

        suggestions = []
        for query, freq in frequencies.most_common(limit):
            suggestions.append(
                SearchSuggestion(
                    query=query,
                    confidence=1.0,
                    frequency=freq,
                    last_used=last_used_map.get(query),
                )
            )

        return suggestions

    def get_recent_queries(self, user_id: str, limit: int = 10) -> list[str]:
        """
        Get most recent unique queries.

        Args:
            user_id: User ID
            limit: Maximum number of queries

        Returns:
            List of query strings (most recent first)
        """
        queries = self._query_history.get(user_id, [])
        if not queries:
            return []

        # Sort by timestamp descending
        sorted_queries = sorted(queries, key=lambda x: x[1], reverse=True)

        # Get unique queries (preserve order)
        seen = set()
        unique_queries = []
        for query, _ in sorted_queries:
            if query not in seen:
                seen.add(query)
                unique_queries.append(query)
                if len(unique_queries) >= limit:
                    break

        return unique_queries

    def clear_history(self, user_id: str, days_to_keep: Optional[int] = None):
        """
        Clear query history for a user.

        Args:
            user_id: User ID
            days_to_keep: If provided, only clear history older than N days
        """
        if days_to_keep is None:
            # Clear all
            if user_id in self._query_history:
                del self._query_history[user_id]
            if user_id in self._query_frequencies:
                del self._query_frequencies[user_id]
            logger.info(f"Cleared all query history for user {user_id}")
        else:
            # Clear old history
            cutoff = datetime.now(timezone.utc) - timedelta(days=days_to_keep)
            if user_id in self._query_history:
                self._query_history[user_id] = [
                    (q, ts) for q, ts in self._query_history[user_id] if ts >= cutoff
                ]

                # Rebuild frequency counter
                self._query_frequencies[user_id] = Counter(
                    q for q, _ in self._query_history[user_id]
                )

            logger.info(f"Cleared query history older than {days_to_keep} days for user {user_id}")

    def _clean_old_history(self, user_id: str):
        """Remove queries older than history_days."""
        if user_id not in self._query_history:
            return

        cutoff = datetime.now(timezone.utc) - timedelta(days=self.history_days)
        original_count = len(self._query_history[user_id])

        self._query_history[user_id] = [
            (q, ts) for q, ts in self._query_history[user_id] if ts >= cutoff
        ]

        # Rebuild frequency counter
        self._query_frequencies[user_id] = Counter(q for q, _ in self._query_history[user_id])

        cleaned = original_count - len(self._query_history[user_id])
        if cleaned > 0:
            logger.debug(f"Cleaned {cleaned} old queries for user {user_id}")

    def get_statistics(self, user_id: str) -> dict:
        """
        Get statistics about query history.

        Returns:
            Dictionary with statistics
        """
        queries = self._query_history.get(user_id, [])
        frequencies = self._query_frequencies.get(user_id, Counter())

        if not queries:
            return {
                "total_queries": 0,
                "unique_queries": 0,
                "avg_frequency": 0.0,
                "most_common": None,
            }

        most_common = frequencies.most_common(1)

        return {
            "total_queries": len(queries),
            "unique_queries": len(frequencies),
            "avg_frequency": len(queries) / max(len(frequencies), 1),
            "most_common": most_common[0] if most_common else None,
        }
