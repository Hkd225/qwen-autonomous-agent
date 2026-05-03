"""
tools/web_search.py - Web Search Tool
======================================
Tool untuk melakukan web search menggunakan DuckDuckGo (free, no API key)
atau SerpAPI (lebih powerful, butuh API key).

Referensi Paper:
- ReAct (Yao et al., 2022): "Search[entity]" sebagai salah satu tool utama
  → Web search adalah tool paling fundamental untuk grounding agent
- SELF-RAG (Asai et al., 2023): Retrieval dari web untuk augment generation
  → Agent memutuskan kapan perlu search vs tidak
- WebGPT (Nakano et al., 2021): Human feedback untuk web browsing
  → Inspirasi untuk search-then-summarize pattern
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from .base_tool import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class WebSearchTool(BaseTool):
    """
    Tool web search menggunakan DuckDuckGo (default) atau SerpAPI.

    DuckDuckGo dipilih sebagai default karena:
    - Gratis, tidak butuh API key
    - Privacy-respecting
    - Hasil yang decent untuk general knowledge

    Fallback ke SerpAPI jika tersedia (lebih akurat, structured results).
    """

    def __init__(
        self,
        max_results: int = 5,
        serpapi_key: Optional[str] = None,
    ):
        self.max_results = max_results
        self.serpapi_key = serpapi_key

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return (
            "Cari informasi di internet menggunakan web search. "
            "Gunakan tool ini ketika kamu butuh informasi terkini, "
            "fakta spesifik, atau konteks yang tidak ada di memorimu. "
            "Input: query pencarian dalam bahasa natural. "
            "Output: ringkasan hasil pencarian dengan judul dan URL."
        )

    @property
    def parameter_schema(self) -> Dict[str, Any]:
        return {
            "query": {
                "type": "string",
                "description": "Query pencarian. Gunakan kata kunci yang spesifik dan relevan.",
                "required": True,
            },
            "num_results": {
                "type": "integer",
                "description": "Jumlah hasil yang diinginkan (default: 5, max: 10)",
                "required": False,
                "default": 5,
            },
        }

    async def execute(self, query: str, num_results: int = 5) -> ToolResult:
        """
        Jalankan web search.

        Mencoba DuckDuckGo terlebih dahulu, fallback ke simulasi jika gagal.
        """
        num_results = min(num_results, 10)

        # Coba DuckDuckGo
        try:
            results = await self._search_duckduckgo(query, num_results)
            if results:
                output = self._format_results(query, results)
                return ToolResult(
                    success=True,
                    output=output,
                    data=results,
                    metadata={"source": "duckduckgo", "num_results": len(results)},
                )
        except Exception as e:
            logger.warning(f"DuckDuckGo search gagal: {e}")

        # Coba SerpAPI jika tersedia
        if self.serpapi_key:
            try:
                results = await self._search_serpapi(query, num_results)
                if results:
                    output = self._format_results(query, results)
                    return ToolResult(
                        success=True,
                        output=output,
                        data=results,
                        metadata={"source": "serpapi", "num_results": len(results)},
                    )
            except Exception as e:
                logger.warning(f"SerpAPI search gagal: {e}")

        # Fallback: informasikan bahwa search tidak tersedia
        return ToolResult(
            success=False,
            output="",
            error=(
                f"Web search untuk '{query}' tidak berhasil. "
                "Install duckduckgo-search: pip install duckduckgo-search"
            ),
        )

    async def _search_duckduckgo(
        self, query: str, num_results: int
    ) -> List[Dict[str, str]]:
        """Search menggunakan duckduckgo-search library."""
        from duckduckgo_search import DDGS

        loop = asyncio.get_event_loop()

        def _search():
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(
                    query,
                    max_results=num_results,
                    region="wt-wt",
                ):
                    results.append({
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", ""),
                    })
            return results

        return await loop.run_in_executor(None, _search)

    async def _search_serpapi(
        self, query: str, num_results: int
    ) -> List[Dict[str, str]]:
        """Search menggunakan SerpAPI."""
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://serpapi.com/search",
                params={
                    "q": query,
                    "num": num_results,
                    "api_key": self.serpapi_key,
                    "engine": "google",
                },
                timeout=15.0,
            )
            response.raise_for_status()
            data = response.json()

        results = []
        for r in data.get("organic_results", [])[:num_results]:
            results.append({
                "title": r.get("title", ""),
                "url": r.get("link", ""),
                "snippet": r.get("snippet", ""),
            })
        return results

    def _format_results(
        self, query: str, results: List[Dict[str, str]]
    ) -> str:
        """Format hasil search sebagai teks yang mudah dibaca LLM."""
        lines = [f"Hasil pencarian untuk: '{query}'\n"]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r.get('title', 'Tanpa Judul')}")
            lines.append(f"   URL: {r.get('url', '')}")
            lines.append(f"   {r.get('snippet', '')}")
            lines.append("")
        return "\n".join(lines)

    @property
    def timeout_seconds(self) -> float:
        return 20.0
