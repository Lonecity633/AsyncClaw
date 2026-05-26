from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from AsyncClaw.channels import AgentService
from AsyncClaw.config import MCPConfig
from AsyncClaw.tools import (
    ToolContext,
    ToolExecutor,
    ToolRegistry,
    build_tool_registry,
    web_fetch_tool,
    web_search_tool,
)


class FakeTavilyClient:
    calls: list[dict[str, object]] = []

    def __init__(self, *, api_key: str) -> None:
        self.call: dict[str, object] = {"api_key": api_key}
        FakeTavilyClient.calls.append(self.call)

    def search(self, **kwargs):
        self.call["search"] = kwargs
        return {"results": [{"title": "AsyncClaw", "url": "https://example.test"}]}

    def extract(self, **kwargs):
        self.call["extract"] = kwargs
        return {
            "results": [
                {
                    "url": kwargs["urls"],
                    "raw_content": "# Example",
                }
            ],
            "failed_results": [],
        }


class SimpleLLM:
    def __init__(self) -> None:
        self.requests = []

    def create_chat_completion(self, **kwargs):
        self.requests.append(kwargs)
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}


class TavilyWebToolsTest(unittest.TestCase):
    def setUp(self) -> None:
        FakeTavilyClient.calls = []

    def fake_tavily_module(self):
        return types.SimpleNamespace(TavilyClient=FakeTavilyClient)

    def test_web_search_calls_tavily_with_search_arguments(self) -> None:
        with patch.dict(sys.modules, {"tavily": self.fake_tavily_module()}):
            with patch.dict(os.environ, {"TAVILY_API_KEY": "tvly-test"}, clear=True):
                result = web_search_tool.call(
                    {
                        "query": "AsyncClaw Tavily",
                        "max_results": 3,
                        "topic": "news",
                        "search_depth": "advanced",
                        "include_domains": ["example.test"],
                    }
                )

        self.assertEqual(result["results"][0]["title"], "AsyncClaw")
        self.assertEqual(FakeTavilyClient.calls[0]["api_key"], "tvly-test")
        self.assertEqual(
            FakeTavilyClient.calls[0]["search"],
            {
                "query": "AsyncClaw Tavily",
                "max_results": 3,
                "search_depth": "advanced",
                "topic": "news",
                "include_domains": ["example.test"],
            },
        )

    def test_web_fetch_calls_tavily_extract_with_default_arguments(self) -> None:
        with patch.dict(sys.modules, {"tavily": self.fake_tavily_module()}):
            with patch.dict(os.environ, {"TAVILY_API_KEY": "tvly-test"}, clear=True):
                result = web_fetch_tool.call({"url": "https://example.test/page"})

        self.assertEqual(result["results"][0]["url"], "https://example.test/page")
        self.assertEqual(
            FakeTavilyClient.calls[0]["extract"],
            {
                "urls": "https://example.test/page",
                "extract_depth": "basic",
                "format": "markdown",
                "include_images": False,
            },
        )

    def test_missing_tavily_api_key_returns_readable_tool_error(self) -> None:
        executor = ToolExecutor(ToolRegistry([web_search_tool]))

        with patch.dict(os.environ, {}, clear=True):
            result = executor.execute("web_search", {"query": "AsyncClaw"})

        self.assertIn("TAVILY_API_KEY", result.result["error"])

    def test_build_tool_registry_includes_web_tools(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = build_tool_registry(
                ToolContext(cwd=Path(directory), allow_shell_exec=False)
            )

        tool_names = [tool["function"]["name"] for tool in registry.to_openai_tools()]
        self.assertIn("web_search", tool_names)
        self.assertIn("web_fetch", tool_names)

    def test_agent_service_exposes_web_tools_without_mcp(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = AgentService(
                cwd=directory,
                llm=SimpleLLM(),
                mcp_config=MCPConfig(),
                allow_shell_exec=False,
            )

        tool_names = [tool["function"]["name"] for tool in service.tools.to_openai_tools()]
        self.assertIn("web_search", tool_names)
        self.assertIn("web_fetch", tool_names)


if __name__ == "__main__":
    unittest.main()
