"""Tests for ClaudeCodeWebSearchHandler."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from litellm_claude_code_websearch import handler as handler_mod
from litellm_claude_code_websearch.handler import (
    ClaudeCodeWebSearchHandler,
    _extract_query,
    _format_results,
)

STANDALONE_TOOLS = [
    {"type": "web_search_20250305", "name": "web_search", "max_uses": 8}
]


def _result(url="https://example.com/a", title="Example A", snippet="A <strong>hit</strong>"):
    return SimpleNamespace(url=url, title=title, snippet=snippet, last_updated=None, date=None)


def _search_response(results):
    return SimpleNamespace(results=results)


@pytest.fixture
def handler():
    return ClaudeCodeWebSearchHandler(enabled_providers=["hosted_vllm"])


@pytest.fixture(autouse=True)
def _reset_day_counter():
    handler_mod._day_counter.clear()
    yield
    handler_mod._day_counter.clear()


async def test_standalone_request_short_circuits_with_native_blocks(handler):
    with patch("litellm.asearch", new_callable=AsyncMock) as mock_search:
        mock_search.return_value = _search_response([_result()])

        response = await handler.try_short_circuit_search(
            model="hosted_vllm/some-model",
            messages=[
                {"role": "user", "content": "Perform a web search for: python releases"}
            ],
            tools=STANDALONE_TOOLS,
            custom_llm_provider="hosted_vllm",
        )

    assert response is not None
    block_types = [b["type"] for b in response["content"]]
    assert block_types == ["server_tool_use", "web_search_tool_result", "text"]
    server_tool_use = response["content"][0]
    assert server_tool_use["input"] == {"query": "python releases"}
    tool_result = response["content"][1]
    assert tool_result["tool_use_id"] == server_tool_use["id"]
    assert tool_result["content"][0]["url"] == "https://example.com/a"
    assert response["usage"]["server_tool_use"] == {"web_search_requests": 1}
    mock_search.assert_awaited_once()


async def test_mixed_tools_pass_through(handler):
    response = await handler.try_short_circuit_search(
        model="hosted_vllm/some-model",
        messages=[{"role": "user", "content": "do things"}],
        tools=STANDALONE_TOOLS
        + [{"name": "Bash", "description": "run", "input_schema": {"type": "object"}}],
        custom_llm_provider="hosted_vllm",
    )

    assert response is None


async def test_disabled_provider_passes_through(handler):
    response = await handler.try_short_circuit_search(
        model="openai/gpt-4o",
        messages=[{"role": "user", "content": "search: x"}],
        tools=STANDALONE_TOOLS,
        custom_llm_provider="openai",
    )

    assert response is None


async def test_pre_hooks_are_neutralized(handler):
    assert await handler.async_pre_request_hook("m", [], {}) is None
    assert await handler.async_pre_call_deployment_hook({}, None) is None


async def test_search_failure_returns_graceful_text(handler):
    with patch("litellm.asearch", new_callable=AsyncMock) as mock_search:
        mock_search.side_effect = RuntimeError("provider down")

        response = await handler.try_short_circuit_search(
            model="hosted_vllm/some-model",
            messages=[{"role": "user", "content": "search: x"}],
            tools=STANDALONE_TOOLS,
            custom_llm_provider="hosted_vllm",
        )

    assert response is not None
    assert response["content"][-1]["type"] == "text"
    assert "Web search unavailable" in response["content"][-1]["text"]


async def test_daily_limit_short_circuits_without_search(handler, monkeypatch):
    monkeypatch.setattr(handler_mod, "_DAILY_LIMIT", 1)
    with patch("litellm.asearch", new_callable=AsyncMock) as mock_search:
        mock_search.return_value = _search_response([_result()])

        first = await handler.try_short_circuit_search(
            model="hosted_vllm/m",
            messages=[{"role": "user", "content": "search: one"}],
            tools=STANDALONE_TOOLS,
            custom_llm_provider="hosted_vllm",
        )
        second = await handler.try_short_circuit_search(
            model="hosted_vllm/m",
            messages=[{"role": "user", "content": "search: two"}],
            tools=STANDALONE_TOOLS,
            custom_llm_provider="hosted_vllm",
        )

    assert first is not None
    assert mock_search.await_count == 1
    assert "daily search limit" in second["content"][-1]["text"]


async def test_blocked_domains_become_negative_site_operators(handler):
    tools = [dict(STANDALONE_TOOLS[0], blocked_domains=["spam.example"])]
    with patch("litellm.asearch", new_callable=AsyncMock) as mock_search:
        mock_search.return_value = _search_response([_result()])

        await handler.try_short_circuit_search(
            model="hosted_vllm/m",
            messages=[{"role": "user", "content": "search: cats"}],
            tools=tools,
            custom_llm_provider="hosted_vllm",
        )

    assert "-site:spam.example" in mock_search.await_args.kwargs["query"]


async def test_allowed_domains_forwarded_as_domain_filter(handler):
    tools = [dict(STANDALONE_TOOLS[0], allowed_domains=["docs.python.org"])]
    with patch("litellm.asearch", new_callable=AsyncMock) as mock_search:
        mock_search.return_value = _search_response([_result()])

        await handler.try_short_circuit_search(
            model="hosted_vllm/m",
            messages=[{"role": "user", "content": "search: cats"}],
            tools=tools,
            custom_llm_provider="hosted_vllm",
        )

    assert mock_search.await_args.kwargs["search_domain_filter"] == ["docs.python.org"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Perform a web search for the query: rust editions", "rust editions"),
        ("Search the web for: 'quoted thing'", "quoted thing"),
        ("web search for llama models", "llama models"),
        ("plain text with no boilerplate", "plain text with no boilerplate"),
    ],
)
def test_extract_query(raw, expected):
    assert _extract_query(raw) == expected


def test_format_results_strips_markup_and_lists_urls():
    text = _format_results("q", [_result()])
    assert "1. Example A" in text
    assert "URL: https://example.com/a" in text
    assert "<strong>" not in text
    assert "A hit" in text


def test_format_results_empty():
    assert "returned no results" in _format_results("q", [])
