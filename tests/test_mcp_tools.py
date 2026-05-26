from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from AsyncClaw.channels import AgentService
from AsyncClaw.config import MCPConfig, MCPServerConfig, load_mcp_config


class MCPCallingLLM:
    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name
        self.requests = []

    def create_chat_completion(self, **kwargs):
        self.requests.append(kwargs)
        if len(self.requests) == 1:
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_mcp",
                                    "type": "function",
                                    "function": {
                                        "name": self.tool_name,
                                        "arguments": json.dumps({"query": "asyncclaw"}),
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "MCP 已调用",
                    }
                }
            ]
        }


class MultiplyLLM:
    def __init__(self) -> None:
        self.requests = []

    def create_chat_completion(self, **kwargs):
        self.requests.append(kwargs)
        if len(self.requests) == 1:
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_multiply",
                                    "type": "function",
                                    "function": {
                                        "name": "multiply",
                                        "arguments": json.dumps({"a": 2, "b": 6}),
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "2 * 6 = 12",
                    }
                }
            ]
        }


class SimpleLLM:
    def __init__(self) -> None:
        self.requests = []

    def create_chat_completion(self, **kwargs):
        self.requests.append(kwargs)
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}


class FakeMCPClient:
    tools: list[dict] = []
    tool_result: dict = {"structuredContent": {"ok": True, "source": "mcp"}}
    instances: list[FakeMCPClient] = []

    def __init__(self, *, url: str, headers: dict | None = None, timeout_seconds: float = 10):
        self.url = url
        self.headers = headers or {}
        self.timeout_seconds = timeout_seconds
        self.initialized = False
        self.tool_calls = []
        FakeMCPClient.instances.append(self)

    def initialize(self) -> None:
        self.initialized = True

    def list_tools(self) -> list[dict]:
        return FakeMCPClient.tools

    def call_tool(self, name: str, arguments: dict) -> dict:
        self.tool_calls.append({"name": name, "arguments": arguments})
        return FakeMCPClient.tool_result


class FailingMCPClient(FakeMCPClient):
    def initialize(self) -> None:
        raise RuntimeError("连接失败")


class MCPToolProviderTest(unittest.TestCase):
    def setUp(self) -> None:
        FakeMCPClient.tools = []
        FakeMCPClient.tool_result = {"structuredContent": {"ok": True, "source": "mcp"}}
        FakeMCPClient.instances = []

    def test_no_mcp_config_keeps_local_tools(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = AgentService(
                cwd=directory,
                llm=SimpleLLM(),
                mcp_config=MCPConfig(),
                allow_shell_exec=False,
            )

        tool_names = [tool["function"]["name"] for tool in service.tools.to_openai_tools()]
        self.assertIn("multiply", tool_names)
        self.assertIn("current_time", tool_names)
        self.assertNotIn("vendor_search", tool_names)

    def test_discovers_and_calls_streamable_http_mcp_tool(self) -> None:
        FakeMCPClient.tools = [
            {
                "name": "search",
                "title": "Search",
                "description": "Search remote data.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            }
        ]
        with patch("AsyncClaw.tools.mcp.StreamableHTTPMCPClient", FakeMCPClient):
            with tempfile.TemporaryDirectory() as directory:
                service = AgentService(
                    cwd=directory,
                    llm=MCPCallingLLM("vendor_search"),
                    mcp_config=MCPConfig(
                        servers=(
                            MCPServerConfig(
                                name="vendor",
                                url="https://vendor.example.com/mcp",
                                timeout_seconds=2,
                            ),
                        )
                    ),
                    allow_shell_exec=False,
                )
                response = service.handle_text("search")

        first_tool_names = [
            tool["function"]["name"]
            for tool in service.llm.requests[0]["tools"]
        ]
        self.assertIn("vendor_search", first_tool_names)
        self.assertTrue(FakeMCPClient.instances[0].initialized)
        self.assertEqual(response.output, "MCP 已调用")
        self.assertEqual(response.observations[0]["result"], {"ok": True, "source": "mcp"})
        self.assertEqual(
            FakeMCPClient.instances[0].tool_calls,
            [{"name": "search", "arguments": {"query": "asyncclaw"}}],
        )

    def test_mcp_connection_failure_does_not_disable_local_tools(self) -> None:
        with patch("AsyncClaw.tools.mcp.StreamableHTTPMCPClient", FailingMCPClient):
            with tempfile.TemporaryDirectory() as directory:
                service = AgentService(
                    cwd=directory,
                    llm=MultiplyLLM(),
                    mcp_config=MCPConfig(
                        servers=(
                            MCPServerConfig(
                                name="broken",
                                url="https://broken.example.com/mcp",
                                timeout_seconds=0.1,
                            ),
                        )
                    ),
                    allow_shell_exec=False,
                )
                response = service.handle_text("2 乘以 6")

        self.assertEqual(response.output, "2 * 6 = 12")
        self.assertEqual(response.observations[0]["result"], {"product": 12})

    def test_default_prefix_avoids_builtin_name_collision(self) -> None:
        FakeMCPClient.tools = [
            {
                "name": "multiply",
                "description": "Remote multiply.",
                "inputSchema": {"type": "object", "properties": {}},
            }
        ]
        with patch("AsyncClaw.tools.mcp.StreamableHTTPMCPClient", FakeMCPClient):
            with tempfile.TemporaryDirectory() as directory:
                service = AgentService(
                    cwd=directory,
                    llm=SimpleLLM(),
                    mcp_config=MCPConfig(
                        servers=(
                            MCPServerConfig(
                                name="vendor",
                                url="https://vendor.example.com/mcp",
                            ),
                        )
                    ),
                    allow_shell_exec=False,
                )

        tool_names = [tool["function"]["name"] for tool in service.tools.to_openai_tools()]
        self.assertIn("multiply", tool_names)
        self.assertIn("vendor_multiply", tool_names)

    def test_disabled_prefix_collision_keeps_builtin_tool(self) -> None:
        FakeMCPClient.tools = [
            {
                "name": "multiply",
                "description": "Remote multiply.",
                "inputSchema": {"type": "object", "properties": {}},
            }
        ]
        with patch("AsyncClaw.tools.mcp.StreamableHTTPMCPClient", FakeMCPClient):
            with tempfile.TemporaryDirectory() as directory:
                service = AgentService(
                    cwd=directory,
                    llm=MultiplyLLM(),
                    mcp_config=MCPConfig(
                        servers=(
                            MCPServerConfig(
                                name="vendor",
                                url="https://vendor.example.com/mcp",
                                tool_prefix=False,
                            ),
                        )
                    ),
                    allow_shell_exec=False,
                )
                response = service.handle_text("2 乘以 6")

        tool_names = [tool["function"]["name"] for tool in service.tools.to_openai_tools()]
        self.assertEqual(tool_names.count("multiply"), 1)
        self.assertEqual(response.observations[0]["result"], {"product": 12})

    def test_cron_agent_reuses_mcp_provider_and_honors_allow_cron(self) -> None:
        FakeMCPClient.tools = [
            {
                "name": "search",
                "description": "Search remote data.",
                "inputSchema": {"type": "object", "properties": {}},
            }
        ]
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        with patch("AsyncClaw.tools.mcp.StreamableHTTPMCPClient", FakeMCPClient):
            with tempfile.TemporaryDirectory() as directory:
                service = AgentService(
                    cwd=directory,
                    llm=SimpleLLM(),
                    mcp_config=MCPConfig(
                        servers=(
                            MCPServerConfig(
                                name="vendor",
                                url="https://vendor.example.com/mcp",
                                allow_cron=False,
                            ),
                        )
                    ),
                    allow_shell_exec=False,
                )
                job = service.cron_store.create_job(
                    name="cron",
                    prompt="cron search",
                    schedule={"type": "at", "run_at": now.isoformat()},
                    action="agent",
                    now=now,
                )
                service.handle_cron_text(job)

        startup_tool_names = [
            tool["function"]["name"]
            for tool in service.tools.to_openai_tools()
        ]
        cron_tool_names = [
            tool["function"]["name"]
            for tool in service.llm.requests[-1]["tools"]
        ]
        self.assertIn("vendor_search", startup_tool_names)
        self.assertNotIn("vendor_search", cron_tool_names)

    def test_mcp_config_ignores_shell_env_when_dotenv_does_not_enable_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text("GITHUB_MCP_TOKEN=dotenv_token\n", encoding="utf-8")

            with patch.dict(
                "os.environ",
                {"MCP_CONFIG": "missing-mcp-config.json"},
                clear=True,
            ):
                config = load_mcp_config(env_file=env_path)

        self.assertEqual(config, MCPConfig())

    def test_dotenv_mcp_config_loads_relative_to_dotenv_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "mcp.servers.json"
            config_path.write_text(
                json.dumps(
                    {
                        "servers": [
                            {
                                "name": "vendor",
                                "url": "https://vendor.example.com/mcp",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            env_path = root / ".env"
            env_path.write_text("MCP_CONFIG=./mcp.servers.json\n", encoding="utf-8")

            with patch.dict("os.environ", {}, clear=True):
                config = load_mcp_config(env_file=env_path)

        self.assertEqual(len(config.servers), 1)
        self.assertEqual(config.servers[0].name, "vendor")
        self.assertEqual(config.servers[0].url, "https://vendor.example.com/mcp")

    def test_dotenv_mcp_config_missing_file_remains_strict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text("MCP_CONFIG=./missing.json\n", encoding="utf-8")

            with patch.dict("os.environ", {}, clear=True):
                with self.assertRaises(FileNotFoundError):
                    load_mcp_config(env_file=env_path)

    def test_github_mcp_example_config_parses_read_only_headers(self) -> None:
        example_config = (
            Path(__file__).resolve().parents[1] / "mcp.servers.github.example.json"
        )
        with tempfile.TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        f"MCP_CONFIG={example_config}",
                        "GITHUB_MCP_TOKEN=ghp_test_token",
                    ]
                ),
                encoding="utf-8",
            )
            with patch.dict("os.environ", {}, clear=True):
                config = load_mcp_config(env_file=env_path)

        self.assertEqual(len(config.servers), 1)
        server = config.servers[0]
        self.assertEqual(server.name, "github")
        self.assertEqual(server.transport, "streamable-http")
        self.assertEqual(server.url, "https://api.githubcopilot.com/mcp/")
        self.assertEqual(server.timeout_seconds, 15)
        self.assertFalse(server.allow_cron)
        self.assertEqual(server.headers["Authorization"], "Bearer ghp_test_token")
        self.assertEqual(server.headers["X-MCP-Toolsets"], "repos,pull_requests")
        self.assertEqual(server.headers["X-MCP-Readonly"], "true")
        self.assertEqual(server.headers["X-MCP-Lockdown"], "false")

    def test_dotenv_values_override_shell_when_expanding_mcp_headers(self) -> None:
        example_config = (
            Path(__file__).resolve().parents[1] / "mcp.servers.github.example.json"
        )
        with tempfile.TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        f"MCP_CONFIG={example_config}",
                        "GITHUB_MCP_TOKEN=dotenv_token",
                    ]
                ),
                encoding="utf-8",
            )
            with patch.dict(
                "os.environ",
                {"GITHUB_MCP_TOKEN": "shell_token"},
                clear=True,
            ):
                config = load_mcp_config(env_file=env_path)

        self.assertEqual(
            config.servers[0].headers["Authorization"],
            "Bearer dotenv_token",
        )


if __name__ == "__main__":
    unittest.main()
