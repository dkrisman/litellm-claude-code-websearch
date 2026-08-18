"""Short-circuit Claude Code's standalone WebSearch requests through a litellm search provider.

Claude Code executes a web search by issuing a STANDALONE /v1/messages request
whose tools list contains ONLY the Anthropic server tool
{"type": "web_search_20250305", "name": "web_search", ...}. Real Anthropic runs
the search server-side; backends behind a litellm proxy (local vLLM, bridged
providers) cannot. Without a handler the request errors or returns nothing,
and WebSearch is dead against the proxy.

litellm ships ``WebSearchInterceptionLogger`` with a short-circuit path built
for exactly this pattern: when EVERY tool in the request is a web-search tool,
it runs the search itself and returns a synthetic Anthropic response without
calling the backend LLM. This subclass builds on that machinery:

1. Enables the short-circuit for the providers you route Claude Code to.
2. Disables the base class's tool-conversion pre-hooks so ordinary main-agent
   requests (which may carry a client-side tool named ``WebSearch``) pass
   through byte-identical.
3. Executes the search via ``litellm.asearch`` with query extraction tuned to
   Claude Code's search-prompt shape, allowed/blocked domain honoring, a
   configurable daily cap, and optional JSONL tap logging.
4. Returns the canonical Anthropic block sequence (``server_tool_use`` →
   ``web_search_tool_result`` → text) so Claude Code's "Did N searches" line
   and result links render.

Configuration (environment variables, read at import):

  CLAUDE_WEBSEARCH_PROVIDERS        comma-separated litellm providers to
                                    intercept (default "hosted_vllm")
  CLAUDE_WEBSEARCH_SEARCH_PROVIDER  litellm search provider (default "brave")
  CLAUDE_WEBSEARCH_COUNT            results per search (default 8)
  CLAUDE_WEBSEARCH_DAILY_LIMIT      searches per UTC day, 0 = uncapped
                                    (default 0)
  CLAUDE_WEBSEARCH_API_KEY_FILE     optional file to load the search API key
                                    from (docker secrets convenience)
  CLAUDE_WEBSEARCH_API_KEY_VAR      env var the key file populates
                                    (default "BRAVE_API_KEY")
  CLAUDE_WEBSEARCH_TAP_FILE         optional JSONL telemetry file
                                    (default unset = disabled)

Register in the proxy config:

  litellm_settings:
    callbacks:
      - litellm_claude_code_websearch.handler_instance
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from typing import Any, Dict, List, Optional

from litellm.integrations.websearch_interception.handler import (
    WebSearchInterceptionLogger,
)
from litellm.integrations.websearch_interception.tools import is_web_search_tool

logger = logging.getLogger("litellm.claude_code_websearch")

_ENABLED_PROVIDERS = [
    p.strip()
    for p in os.environ.get("CLAUDE_WEBSEARCH_PROVIDERS", "hosted_vllm").split(",")
    if p.strip()
]
_SEARCH_PROVIDER = os.environ.get("CLAUDE_WEBSEARCH_SEARCH_PROVIDER", "brave")
_SEARCH_COUNT = int(os.environ.get("CLAUDE_WEBSEARCH_COUNT", "8"))
_DAILY_LIMIT = int(os.environ.get("CLAUDE_WEBSEARCH_DAILY_LIMIT", "0"))
_TAP_FILE = os.environ.get("CLAUDE_WEBSEARCH_TAP_FILE", "")
_QUERY_MAX_CHARS = 400  # Brave rejects queries > 400 chars / 50 words
_SNIPPET_CAP = 400

# Docker-secrets convenience: load the search API key from a file when the
# target env var is not already set. Env always wins.
_KEY_FILE = os.environ.get("CLAUDE_WEBSEARCH_API_KEY_FILE", "")
_KEY_VAR = os.environ.get("CLAUDE_WEBSEARCH_API_KEY_VAR", "BRAVE_API_KEY")
if _KEY_FILE and not os.environ.get(_KEY_VAR):
    try:
        with open(_KEY_FILE) as _f:
            _key = _f.read().strip()
        if _key:
            os.environ[_KEY_VAR] = _key
            logger.info("claude_code_websearch: %s loaded from %s", _KEY_VAR, _KEY_FILE)
    except OSError:
        logger.warning(
            "claude_code_websearch: %s unset and key file %s unreadable; "
            "searches will fail until a key is provided",
            _KEY_VAR,
            _KEY_FILE,
        )

# In-process daily counter: a budgetary signal, not a hard guarantee.
# Resets on process restart, which is acceptable for its purpose.
_day_counter: Dict[str, int] = {}


def _day_key() -> str:
    return time.strftime("%Y-%m-%d", time.gmtime())


def _tap(obj: Dict[str, Any]) -> None:
    """Best-effort JSONL telemetry; never raises into the request path."""
    if not _TAP_FILE:
        return
    try:
        parent = os.path.dirname(_TAP_FILE)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(_TAP_FILE, "a") as f:
            f.write(json.dumps(obj, default=str) + "\n")
    except Exception:
        pass


_TAG_RE = re.compile(r"<[^>]{1,80}>")

# Claude Code wraps the query in an instruction sentence whose exact wording
# varies by version. Fallback: the whole message is the query.
_QUERY_PATTERNS = [
    re.compile(r"(?is)perform a web search for(?: the)?(?: query)?:?\s*(.+)"),
    re.compile(r"(?is)^web search(?: results?)? for(?: the)?(?: query)?:?\s*(.+)"),
    re.compile(r"(?is)^search the web for:?\s*(.+)"),
    re.compile(r"(?is)^search(?: query)?:\s*(.+)"),
]


def _extract_query(raw: str) -> str:
    """Pull the actual query out of the harness's instruction boilerplate."""
    text = raw.strip()
    for pat in _QUERY_PATTERNS:
        m = pat.match(text)
        if m:
            text = m.group(1).strip()
            break
    if len(text) >= 2 and text[0] in "\"'“" and text[-1] in "\"'”":
        text = text[1:-1].strip()
    return text[:_QUERY_MAX_CHARS]


def _clean_snippet(s: str) -> str:
    """Search snippets may carry HTML markup; strip tags and common entities."""
    s = _TAG_RE.sub("", s or "")
    for ent, ch in (
        ("&amp;", "&"),
        ("&lt;", "<"),
        ("&gt;", ">"),
        ("&quot;", '"'),
        ("&#x27;", "'"),
        ("&#39;", "'"),
    ):
        s = s.replace(ent, ch)
    return " ".join(s.split())[:_SNIPPET_CAP]


def _format_results(query: str, results: List[Any]) -> str:
    """Synthesize the text block the harness feeds back to the main model."""
    if not results:
        return f'Web search for "{query}" returned no results. Try a different query.'
    lines = [f'Web search results for query: "{query}"', ""]
    for i, r in enumerate(results, 1):
        title = getattr(r, "title", "") or "(untitled)"
        url = getattr(r, "url", "") or ""
        snippet = _clean_snippet(getattr(r, "snippet", "") or "")
        date = getattr(r, "last_updated", None) or getattr(r, "date", None)
        lines.append(f"{i}. {title}")
        lines.append(f"   URL: {url}")
        if snippet:
            lines.append(f"   {snippet}")
        if date:
            lines.append(f"   (as of {date})")
        lines.append("")
    lines.append("Use WebFetch on a URL above to read a full page.")
    return "\n".join(lines)


def _synthetic_response(
    model: str,
    text: str,
    query: Optional[str] = None,
    results: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """Anthropic-shaped message the short-circuit path returns.

    When search results are present, emit the canonical Anthropic server-tool
    block sequence (server_tool_use → web_search_tool_result → text), not just
    a bare text block: Claude Code counts web_search_tool_result blocks for its
    "Did N searches" line and renders result links from them. encrypted_content
    is an opaque placeholder; clients never decrypt it, it only round-trips.
    """
    content: List[Dict[str, Any]] = []
    n_searches = 0
    if query is not None and results:
        tool_use_id = f"srvtoolu_{uuid.uuid4().hex}"
        content.append(
            {
                "type": "server_tool_use",
                "id": tool_use_id,
                "name": "web_search",
                "input": {"query": query},
            }
        )
        content.append(
            {
                "type": "web_search_tool_result",
                "tool_use_id": tool_use_id,
                "content": [
                    {
                        "type": "web_search_result",
                        "url": getattr(r, "url", "") or "",
                        "title": getattr(r, "title", "") or "(untitled)",
                        "page_age": (
                            getattr(r, "last_updated", None) or getattr(r, "date", None)
                        ),
                        "encrypted_content": "bG9jYWwtc3ludGhlc2lzOnYx",
                    }
                    for r in results
                ],
            }
        )
        n_searches = 1
    content.append({"type": "text", "text": text})
    return {
        "id": f"msg_{uuid.uuid4()}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 0,
            "output_tokens": 0,
            "server_tool_use": {"web_search_requests": n_searches},
        },
    }


class ClaudeCodeWebSearchHandler(WebSearchInterceptionLogger):
    """Short-circuit-only WebSearch handler for the Claude Code standalone
    request pattern. Inherits the base class so litellm's
    ``_try_websearch_short_circuit`` picks it up via isinstance; overrides the
    pieces whose stock behavior would hurt a coding-agent harness."""

    def __init__(self, enabled_providers: Optional[List[str]] = None) -> None:
        super().__init__(
            enabled_providers=enabled_providers or _ENABLED_PROVIDERS,
        )

    # The stock pre-hooks rewrite ANY request whose tools match
    # is_web_search_tool, renaming the tool and flipping stream off. Main-agent
    # requests must pass through untouched; the harness executes WebSearch by
    # making the standalone request the short-circuit below handles.
    async def async_pre_request_hook(self, model, messages, kwargs):  # type: ignore[override]
        return None

    async def async_pre_call_deployment_hook(self, kwargs, call_type):  # type: ignore[override]
        return None

    async def try_short_circuit_search(  # type: ignore[override]
        self,
        model: str,
        messages: List[Dict],
        tools: Optional[List[Dict]],
        custom_llm_provider: Optional[str],
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not tools:
            return None
        provider = custom_llm_provider or ""
        if provider not in self.enabled_providers:
            return None
        # Standalone WebSearch request = EVERY tool is a web-search tool.
        # Main-agent requests carry other tools too and never match here.
        if not all(is_web_search_tool(t) for t in tools):
            return None

        from litellm.litellm_core_utils.prompt_templates.common_utils import (
            get_last_user_message,
        )

        raw = get_last_user_message(messages) or ""
        if not raw.strip():
            return None
        query = _extract_query(raw)

        tool_cfg = tools[0] if isinstance(tools[0], dict) else {}
        allowed = tool_cfg.get("allowed_domains") or None
        blocked = tool_cfg.get("blocked_domains") or None
        if blocked:
            # Brave has no blocklist param; negative site: operators emulate it.
            neg = " ".join(f"-site:{d}" for d in blocked if isinstance(d, str))
            query = f"{query} {neg}"[:_QUERY_MAX_CHARS]

        t0 = time.time()
        entry: Dict[str, Any] = {
            "t": t0,
            "model": model,
            "provider": provider,
            "raw_preview": raw[:300],
            "query": query,
            **({"allowed_domains": allowed} if allowed else {}),
            **({"blocked_domains": blocked} if blocked else {}),
        }

        # Daily cap: skip the search round-trip and return a graceful error
        # text. HTTP 200, so the agent continues without results.
        day = _day_key()
        used = _day_counter.get(day, 0)
        if _DAILY_LIMIT > 0 and used >= _DAILY_LIMIT:
            _tap({**entry, "error": "daily_limit", "used": used})
            return _synthetic_response(
                model,
                f"Web search unavailable: daily search limit "
                f"({_DAILY_LIMIT}) reached. Continue without search results.",
            )
        _day_counter.clear()  # keep only the current day
        _day_counter[day] = used + 1

        try:
            import litellm

            result = await litellm.asearch(
                query=query,
                search_provider=_SEARCH_PROVIDER,
                max_results=_SEARCH_COUNT,
                search_domain_filter=allowed,
            )
            results = list(getattr(result, "results", []) or [])
            text = _format_results(query, results)
            _tap(
                {
                    **entry,
                    "n_results": len(results),
                    "ms": int((time.time() - t0) * 1000),
                    "day_used": used + 1,
                }
            )
            return _synthetic_response(model, text, query=query, results=results)
        except Exception as e:
            # Never raise into the request path and never fall through to the
            # backend LLM; it would choke on the server-tool-only request.
            logger.warning("claude_code_websearch: search failed: %s", e)
            _tap({**entry, "error": str(e)[:500], "ms": int((time.time() - t0) * 1000)})
            return _synthetic_response(
                model,
                f"Web search unavailable ({type(e).__name__}). "
                "Continue without search results.",
            )


handler_instance = ClaudeCodeWebSearchHandler()
