"""Query router for typed collections."""

import re
from typing import Literal


class QueryRouter:
    """Routes queries to appropriate collections based on keywords."""

    PREF_KEYWORDS = {
        "prefer",
        "like",
        "love",
        "hate",
        "dislike",
        "want",
        "need",
        "goal",
        "wish",
        "hope",
        "usually",
        "always",
        "never",
        "habit",
        "routine",
    }

    FACT_KEYWORDS = {
        "live",
        "work",
        "born",
        "studied",
        "graduated",
        "married",
        "located",
        "happened",
        "event",
        "visited",
        "traveled",
        "bought",
        "sold",
    }

    @staticmethod
    def _matches_keywords(words: set[str], keywords: set[str]) -> bool:
        """Check if any query word starts with (or equals) a keyword stem, or vice versa.

        This handles plural/conjugated forms: "habits"→"habit", "preferences"→"prefer",
        "working"→"work", "lived"→"live", etc.
        """
        for word in words:
            for kw in keywords:
                if word == kw or word.startswith(kw) or kw.startswith(word):
                    return True
        return False

    def route(self, query: str) -> Literal["prefs", "facts", "both"]:
        """
        Determine which collection(s) to search.

        Returns:
            "prefs", "facts", or "both"
        """
        query_lower = query.lower()
        words = set(re.findall(r"\b\w+\b", query_lower))

        has_pref = self._matches_keywords(words, self.PREF_KEYWORDS)
        has_fact = self._matches_keywords(words, self.FACT_KEYWORDS)

        if has_pref and not has_fact:
            return "prefs"
        if has_fact and not has_pref:
            return "facts"
        return "both"
