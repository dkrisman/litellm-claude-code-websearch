# litellm-claude-code-websearch

A [LiteLLM proxy](https://github.com/BerriAI/litellm) callback that makes Claude Code's
**WebSearch** tool work against backends that can't run web search server-side
(local vLLM, bridged providers), using any LiteLLM search provider — Brave by default.

## How it works

Claude Code executes a web search by sending a **standalone** `/v1/messages` request whose
tools list contains only the Anthropic server tool
`{"type": "web_search_20250305", "name": "web_search", ...}`. Real Anthropic runs the search
server-side; your proxied backend can't, so without this handler the request errors and
WebSearch is dead.

This package subclasses LiteLLM's own `WebSearchInterceptionLogger` short-circuit machinery:

- **Standalone search requests** (every tool is a web-search tool) never reach the backend
  LLM. The handler extracts the query from Claude Code's prompt boilerplate, runs it through
  `litellm.asearch`, and returns a synthetic Anthropic response with the canonical block
  sequence `server_tool_use` → `web_search_tool_result` → text, so Claude Code's
  "Did N searches" line and result links render correctly.
- **Main-agent requests** (which carry Bash/Read/... alongside, and may include a client-side
  tool named `WebSearch`) pass through byte-identical: the base class's tool-conversion
  pre-hooks are disabled.
- `allowed_domains` is forwarded to the search provider; `blocked_domains` is emulated with
  negative `site:` operators. Failures and a configurable daily cap degrade gracefully to an
  HTTP 200 "search unavailable" text so the agent continues.

## Install

```bash
pip install litellm-claude-code-websearch
```

## Configure

`proxy_config.yaml`:

```yaml
model_list:
  - model_name: my-model
    litellm_params:
      model: hosted_vllm/my-model
      api_base: http://localhost:8000/v1
      api_key: dummy

search_tools:
  - search_tool_name: brave-search
    litellm_params:
      search_provider: brave

litellm_settings:
  callbacks:
    - litellm_claude_code_websearch.handler_instance
```

Environment variables (all optional):

| Variable | Default | Purpose |
|---|---|---|
| `CLAUDE_WEBSEARCH_PROVIDERS` | `hosted_vllm` | comma-separated LiteLLM providers to intercept |
| `CLAUDE_WEBSEARCH_SEARCH_PROVIDER` | `brave` | LiteLLM search provider to execute with |
| `CLAUDE_WEBSEARCH_COUNT` | `8` | results per search |
| `CLAUDE_WEBSEARCH_DAILY_LIMIT` | `0` (uncapped) | searches per UTC day, graceful refusal past the cap |
| `CLAUDE_WEBSEARCH_API_KEY_FILE` | unset | file to load the search API key from (docker secrets) |
| `CLAUDE_WEBSEARCH_API_KEY_VAR` | `BRAVE_API_KEY` | env var the key file populates |
| `CLAUDE_WEBSEARCH_TAP_FILE` | unset (off) | JSONL telemetry file for search requests |

The search provider's API key is read by LiteLLM itself (`BRAVE_API_KEY` for Brave). The
`_API_KEY_FILE` knob is a convenience that populates that env var from a mounted secrets file
at import time; the env var always wins if both are set.

## Requirements and caveats

- LiteLLM ≥ 1.96 (needs `WebSearchInterceptionLogger.try_short_circuit_search`).
- Streaming clients receive the `server_tool_use` / `web_search_tool_result` blocks only on
  LiteLLM versions that include
  [BerriAI/litellm#37318](https://github.com/BerriAI/litellm/pull/37318); on older versions
  the fake-stream re-wrap drops those blocks (the text results still arrive, so searches keep
  working, but Claude Code shows "Did 0 searches").
- The daily cap is an in-process counter: it resets on restart and is a budgetary signal, not
  a hard guarantee.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
