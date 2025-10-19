"""Tools for AI assistant - web search, calculations, etc.

This module provides tool calling capabilities for the AI assistant,
including web search, calculations, and more.
"""

import logging
import json
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime
from abc import ABC, abstractmethod


logger = logging.getLogger(__name__)


class BaseTool(ABC):
    """Base class for all tools."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool with given parameters."""
        pass

    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema for LLM function calling."""
        pass


class WebSearchTool(BaseTool):
    """
    Web search tool using DuckDuckGo, Google, or Brave Search.

    Supports multiple search engines with automatic fallback.
    """

    def __init__(self, search_engine: str = "duckduckgo"):
        """
        Initialize web search tool.

        Args:
            search_engine: Engine to use (duckduckgo, google, brave)
        """
        super().__init__(
            name="web_search",
            description="Search the web for current information, news, facts, and answers to questions"
        )
        self.search_engine = search_engine.lower()
        logger.info(f"WebSearchTool initialized with engine: {self.search_engine}")

    def execute(self, query: str, num_results: int = 5) -> Dict[str, Any]:
        """
        Execute web search.

        Args:
            query: Search query
            num_results: Number of results to return (default: 5)

        Returns:
            Dictionary with search results
        """
        logger.info(f"Searching web for: {query}")

        try:
            if self.search_engine == "duckduckgo":
                results = self._search_duckduckgo(query, num_results)
            elif self.search_engine == "google":
                results = self._search_google(query, num_results)
            elif self.search_engine == "brave":
                results = self._search_brave(query, num_results)
            else:
                # Default to DuckDuckGo
                results = self._search_duckduckgo(query, num_results)

            logger.info(f"Found {len(results)} search results")

            return {
                "success": True,
                "query": query,
                "engine": self.search_engine,
                "results": results,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {
                "success": False,
                "query": query,
                "error": str(e),
                "results": []
            }

    def _search_duckduckgo(self, query: str, num_results: int) -> List[Dict[str, str]]:
        """
        Search using DuckDuckGo Instant Answer API.

        This is free and doesn't require API keys.
        """
        try:
            # DuckDuckGo Instant Answer API
            url = "https://api.duckduckgo.com/"
            params = {
                "q": query,
                "format": "json",
                "no_html": 1,
                "skip_disambig": 1
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            results = []

            # Add abstract if available
            if data.get("Abstract"):
                results.append({
                    "title": data.get("Heading", query),
                    "snippet": data.get("Abstract"),
                    "url": data.get("AbstractURL", ""),
                    "source": data.get("AbstractSource", "DuckDuckGo")
                })

            # Add related topics
            for topic in data.get("RelatedTopics", [])[:num_results - 1]:
                if isinstance(topic, dict) and topic.get("Text"):
                    results.append({
                        "title": topic.get("FirstURL", "").split("/")[-1].replace("_", " "),
                        "snippet": topic.get("Text"),
                        "url": topic.get("FirstURL", ""),
                        "source": "DuckDuckGo"
                    })

            # If no results, try HTML scraping (fallback)
            if not results:
                results = self._search_duckduckgo_html(query, num_results)

            return results[:num_results]

        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
            # Try HTML fallback
            try:
                return self._search_duckduckgo_html(query, num_results)
            except:
                return []

    def _search_duckduckgo_html(self, query: str, num_results: int) -> List[Dict[str, str]]:
        """Fallback: Search DuckDuckGo HTML (simple scraping)."""
        try:
            import re

            url = "https://html.duckduckgo.com/html/"
            params = {"q": query}
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }

            response = requests.post(url, data=params, headers=headers, timeout=10)
            response.raise_for_status()

            html = response.text
            results = []

            # Parse result blocks - DuckDuckGo HTML has a predictable structure
            # Find all result divs
            result_pattern = r'<div class="result__body">.*?<a.*?class="result__a".*?href="(.*?)".*?>(.*?)</a>.*?<a.*?class="result__snippet".*?>(.*?)</a>'
            matches = re.findall(result_pattern, html, re.DOTALL)

            for match in matches[:num_results]:
                url_match, title, snippet = match

                # Clean HTML tags
                title = re.sub(r'<.*?>', '', title).strip()
                snippet = re.sub(r'<.*?>', '', snippet).strip()

                # Decode HTML entities
                import html as html_module
                title = html_module.unescape(title)
                snippet = html_module.unescape(snippet)

                if title and url_match:
                    results.append({
                        "title": title,
                        "snippet": snippet if snippet else "No description available",
                        "url": url_match,
                        "source": "DuckDuckGo"
                    })

            # If regex parsing failed, try simpler approach
            if not results:
                # Extract just links and titles as fallback
                link_pattern = r'<a.*?class="result__a".*?href="(.*?)".*?>(.*?)</a>'
                links = re.findall(link_pattern, html, re.DOTALL)

                for url_match, title in links[:num_results]:
                    title = re.sub(r'<.*?>', '', title).strip()
                    import html as html_module
                    title = html_module.unescape(title)

                    if title and url_match:
                        results.append({
                            "title": title,
                            "snippet": f"Information about {query}",
                            "url": url_match,
                            "source": "DuckDuckGo"
                        })

            return results if results else [{
                "title": f"Search: {query}",
                "snippet": f"Visit DuckDuckGo to search for '{query}'",
                "url": f"https://duckduckgo.com/?q={query.replace(' ', '+')}",
                "source": "DuckDuckGo"
            }]

        except Exception as e:
            logger.error(f"DuckDuckGo HTML search failed: {e}")
            return []

    def _search_google(self, query: str, num_results: int) -> List[Dict[str, str]]:
        """
        Search using Google Custom Search API.

        Requires GOOGLE_API_KEY and GOOGLE_CX in environment.
        """
        import os

        api_key = os.getenv("GOOGLE_API_KEY")
        cx = os.getenv("GOOGLE_CX")

        if not api_key or not cx:
            logger.warning("Google API credentials not found, falling back to DuckDuckGo")
            return self._search_duckduckgo(query, num_results)

        try:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": api_key,
                "cx": cx,
                "q": query,
                "num": num_results
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            results = []
            for item in data.get("items", []):
                results.append({
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "url": item.get("link", ""),
                    "source": "Google"
                })

            return results

        except Exception as e:
            logger.error(f"Google search failed: {e}")
            return self._search_duckduckgo(query, num_results)

    def _search_brave(self, query: str, num_results: int) -> List[Dict[str, str]]:
        """
        Search using Brave Search API.

        Requires BRAVE_API_KEY in environment.
        """
        import os

        api_key = os.getenv("BRAVE_API_KEY")

        if not api_key:
            logger.warning("Brave API key not found, falling back to DuckDuckGo")
            return self._search_duckduckgo(query, num_results)

        try:
            url = "https://api.search.brave.com/res/v1/web/search"
            headers = {
                "X-Subscription-Token": api_key,
                "Accept": "application/json"
            }
            params = {
                "q": query,
                "count": num_results
            }

            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            results = []
            for item in data.get("web", {}).get("results", []):
                results.append({
                    "title": item.get("title", ""),
                    "snippet": item.get("description", ""),
                    "url": item.get("url", ""),
                    "source": "Brave"
                })

            return results

        except Exception as e:
            logger.error(f"Brave search failed: {e}")
            return self._search_duckduckgo(query, num_results)

    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema for LLM function calling."""
        return {
            "name": "web_search",
            "description": "Search the web for current information, news, facts, and answers to questions. Use this when you need up-to-date information, current events, or information you don't have in your training data.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up on the web"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of search results to return (default: 5, max: 10)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }


class CalculatorTool(BaseTool):
    """Simple calculator tool for mathematical operations."""

    def __init__(self):
        super().__init__(
            name="calculator",
            description="Perform mathematical calculations"
        )

    def execute(self, expression: str) -> Dict[str, Any]:
        """
        Execute mathematical expression.

        Args:
            expression: Mathematical expression to evaluate

        Returns:
            Dictionary with calculation result
        """
        try:
            # Safe eval with limited scope
            allowed_names = {
                "abs": abs,
                "round": round,
                "min": min,
                "max": max,
                "sum": sum,
                "pow": pow
            }

            # Evaluate expression
            result = eval(expression, {"__builtins__": {}}, allowed_names)

            return {
                "success": True,
                "expression": expression,
                "result": result
            }

        except Exception as e:
            logger.error(f"Calculation failed: {e}")
            return {
                "success": False,
                "expression": expression,
                "error": str(e)
            }

    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema."""
        return {
            "name": "calculator",
            "description": "Perform mathematical calculations. Supports basic arithmetic, functions like abs, round, min, max, sum, pow.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate (e.g., '2 + 2', 'pow(3, 2)', 'max(10, 20)')"
                    }
                },
                "required": ["expression"]
            }
        }


class ToolRegistry:
    """Registry for managing available tools."""

    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        logger.info("ToolRegistry initialized")

    def register_tool(self, tool: BaseTool):
        """Register a tool."""
        self.tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get tool by name."""
        return self.tools.get(name)

    def get_all_tools(self) -> List[BaseTool]:
        """Get all registered tools."""
        return list(self.tools.values())

    def get_tools_schema(self) -> List[Dict[str, Any]]:
        """Get schemas for all tools (for LLM function calling)."""
        return [tool.get_schema() for tool in self.tools.values()]

    def execute_tool(self, name: str, **kwargs) -> Dict[str, Any]:
        """Execute a tool by name."""
        tool = self.get_tool(name)
        if not tool:
            return {
                "success": False,
                "error": f"Tool '{name}' not found"
            }

        try:
            return tool.execute(**kwargs)
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }


def create_default_registry() -> ToolRegistry:
    """Create registry with default tools."""
    registry = ToolRegistry()

    # Register web search (DuckDuckGo by default, free and no API key needed)
    registry.register_tool(WebSearchTool(search_engine="duckduckgo"))

    # Register calculator
    registry.register_tool(CalculatorTool())

    return registry
