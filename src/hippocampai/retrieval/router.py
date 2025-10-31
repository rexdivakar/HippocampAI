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

    def route(self, query: str) -> Literal["prefs", "facts", "both"]:
        """
        Determine which collection(s) to search.

        Returns:
            "prefs", "facts", or "both"
        """
        query_lower = query.lower()
        words = set(re.findall(r"\b\w+\b", query_lower))

        has_pref = bool(words & self.PREF_KEYWORDS)
        has_fact = bool(words & self.FACT_KEYWORDS)

        if has_pref and not has_fact:
            return "prefs"
        if has_fact and not has_pref:
            return "facts"
        return "both"
