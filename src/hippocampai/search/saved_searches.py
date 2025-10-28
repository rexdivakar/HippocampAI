"""Saved searches manager for quick retrieval."""

import logging
from typing import Optional

from hippocampai.models.search import SavedSearch, SearchMode

logger = logging.getLogger(__name__)


class SavedSearchManager:
    """Manage saved searches for users."""

    def __init__(self):
        """Initialize saved search manager."""
        self._searches: dict[str, SavedSearch] = {}  # search_id -> SavedSearch
        self._user_searches: dict[str, list[str]] = {}  # user_id -> [search_ids]

    def save_search(
        self,
        name: str,
        query: str,
        user_id: str,
        search_mode: SearchMode = SearchMode.HYBRID,
        enable_reranking: bool = True,
        filters: Optional[dict] = None,
        k: int = 5,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict] = None,
    ) -> SavedSearch:
        """
        Save a search query for quick retrieval.

        Args:
            name: Name for this saved search
            query: Search query text
            user_id: User ID
            search_mode: Search mode (hybrid, vector_only, keyword_only)
            enable_reranking: Enable cross-encoder reranking
            filters: Optional filters to apply
            k: Number of results to return
            tags: Optional tags for categorization
            metadata: Optional metadata

        Returns:
            SavedSearch object
        """
        search = SavedSearch(
            name=name,
            query=query,
            user_id=user_id,
            search_mode=search_mode,
            enable_reranking=enable_reranking,
            filters=filters or {},
            k=k,
            tags=tags or [],
            metadata=metadata or {},
        )

        self._searches[search.id] = search

        if user_id not in self._user_searches:
            self._user_searches[user_id] = []
        self._user_searches[user_id].append(search.id)

        logger.info(f"Saved search '{name}' for user {user_id}")
        return search

    def get_search(self, search_id: str) -> Optional[SavedSearch]:
        """Get a saved search by ID."""
        return self._searches.get(search_id)

    def get_user_searches(
        self, user_id: str, tags: Optional[list[str]] = None
    ) -> list[SavedSearch]:
        """
        Get all saved searches for a user.

        Args:
            user_id: User ID
            tags: Optional tags to filter by

        Returns:
            List of SavedSearch objects
        """
        search_ids = self._user_searches.get(user_id, [])
        searches = [self._searches[sid] for sid in search_ids if sid in self._searches]

        if tags:
            tag_set = set(tags)
            searches = [s for s in searches if any(t in tag_set for t in s.tags)]

        # Sort by last used (most recent first), then by use count
        searches.sort(key=lambda s: (s.last_used_at or s.created_at, s.use_count), reverse=True)
        return searches

    def update_search(
        self,
        search_id: str,
        name: Optional[str] = None,
        query: Optional[str] = None,
        search_mode: Optional[SearchMode] = None,
        enable_reranking: Optional[bool] = None,
        filters: Optional[dict] = None,
        k: Optional[int] = None,
        tags: Optional[list[str]] = None,
    ) -> Optional[SavedSearch]:
        """
        Update a saved search.

        Returns:
            Updated SavedSearch or None if not found
        """
        search = self._searches.get(search_id)
        if not search:
            return None

        if name is not None:
            search.name = name
        if query is not None:
            search.query = query
        if search_mode is not None:
            search.search_mode = search_mode
        if enable_reranking is not None:
            search.enable_reranking = enable_reranking
        if filters is not None:
            search.filters = filters
        if k is not None:
            search.k = k
        if tags is not None:
            search.tags = tags

        logger.debug(f"Updated saved search {search_id}")
        return search

    def delete_search(self, search_id: str, user_id: str) -> bool:
        """
        Delete a saved search.

        Args:
            search_id: Search ID
            user_id: User ID (for authorization)

        Returns:
            True if deleted, False if not found or unauthorized
        """
        search = self._searches.get(search_id)
        if not search or search.user_id != user_id:
            return False

        del self._searches[search_id]

        if user_id in self._user_searches:
            self._user_searches[user_id] = [
                sid for sid in self._user_searches[user_id] if sid != search_id
            ]

        logger.info(f"Deleted saved search {search_id}")
        return True

    def execute_saved_search(self, search_id: str) -> Optional[SavedSearch]:
        """
        Get a saved search and mark it as used.

        Returns:
            SavedSearch or None if not found
        """
        search = self._searches.get(search_id)
        if search:
            search.increment_usage()
            logger.debug(
                f"Executed saved search '{search.name}' (use count: {search.use_count})"
            )
        return search

    def get_most_used(self, user_id: str, limit: int = 10) -> list[SavedSearch]:
        """
        Get most frequently used saved searches.

        Args:
            user_id: User ID
            limit: Maximum number to return

        Returns:
            List of SavedSearch objects sorted by use count
        """
        searches = self.get_user_searches(user_id)
        searches.sort(key=lambda s: s.use_count, reverse=True)
        return searches[:limit]

    def search_by_name(self, user_id: str, name_query: str) -> list[SavedSearch]:
        """
        Search saved searches by name.

        Args:
            user_id: User ID
            name_query: Name search query (case-insensitive substring match)

        Returns:
            List of matching SavedSearch objects
        """
        searches = self.get_user_searches(user_id)
        query_lower = name_query.lower()
        return [s for s in searches if query_lower in s.name.lower()]

    def get_statistics(self, user_id: Optional[str] = None) -> dict:
        """
        Get usage statistics for saved searches.

        Args:
            user_id: Optional user ID to filter by

        Returns:
            Dictionary with statistics
        """
        if user_id:
            searches = self.get_user_searches(user_id)
        else:
            searches = list(self._searches.values())

        if not searches:
            return {
                "total_searches": 0,
                "total_uses": 0,
                "avg_uses_per_search": 0.0,
                "most_used_search": None,
            }

        total_uses = sum(s.use_count for s in searches)
        most_used = max(searches, key=lambda s: s.use_count)

        return {
            "total_searches": len(searches),
            "total_uses": total_uses,
            "avg_uses_per_search": total_uses / len(searches),
            "most_used_search": {
                "name": most_used.name,
                "query": most_used.query,
                "use_count": most_used.use_count,
            },
        }
