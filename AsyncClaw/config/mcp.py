"""MCP server configuration loading and parsing."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from AsyncClaw.config.paths import resolve_dotenv_relative_path

MCP_CONFIG_ENV = "MCP_CONFIG"


@dataclass(frozen=True)
class MCPServerConfig:
    """单个 MCP 工具服务配置。"""

    name: str
    transport: str = "streamable-http"
    url: str | None = None
    command: str | None = None
    args: tuple[str, ...] = ()
    headers: dict[str, str] | None = None
    env: dict[str, str] | None = None
    timeout_seconds: float = 10.0
    enabled: bool = True
    tool_prefix: str | bool | None = None
    allow_cron: bool = True


@dataclass(frozen=True)
class MCPConfig:
    """MCP 工具服务集合配置。"""

    servers: tuple[MCPServerConfig, ...] = ()


def load_mcp_config(env_file: str | Path = ".env", override: bool = False) -> MCPConfig:
    """从 `.env` 显式指向的 JSON 文件加载 MCP 工具服务配置。"""

    del override
    try:
        from dotenv import dotenv_values
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "加载 .env 文件需要 python-dotenv。请使用 `pip install -e .` "
            "安装项目依赖。"
        ) from exc

    env_path = Path(env_file)
    env_values = _normalize_dotenv_values(dotenv_values(dotenv_path=env_path))
    config_path = env_values.get(MCP_CONFIG_ENV)
    if not config_path:
        return MCPConfig()

    path = resolve_dotenv_relative_path(config_path, env_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_servers = payload if isinstance(payload, list) else payload.get("servers", [])
    if not isinstance(raw_servers, list):
        raise ValueError("MCP_CONFIG 中的 servers 必须是数组")

    servers = tuple(
        _parse_mcp_server(index, item, env_values)
        for index, item in enumerate(raw_servers)
    )
    return MCPConfig(servers=servers)


def _parse_mcp_server(
    index: int,
    raw: Any,
    env_values: Mapping[str, str] | None = None,
) -> MCPServerConfig:
    if not isinstance(raw, dict):
        raise ValueError(f"MCP server #{index + 1} 必须是对象")

    name = str(raw.get("name") or "").strip()
    if not name:
        raise ValueError(f"MCP server #{index + 1} 缺少 name")

    transport = str(raw.get("transport") or "streamable-http").strip()
    headers = _expand_mapping(raw.get("headers") or {}, env_values)
    env = _expand_mapping(raw.get("env") or {}, env_values)
    timeout = raw.get("timeout_seconds", raw.get("timeout", 10.0))

    return MCPServerConfig(
        name=name,
        transport=transport,
        url=raw.get("url"),
        command=raw.get("command"),
        args=tuple(str(value) for value in raw.get("args") or ()),
        headers=headers,
        env=env,
        timeout_seconds=float(timeout),
        enabled=bool(raw.get("enabled", True)),
        tool_prefix=raw.get("tool_prefix"),
        allow_cron=bool(raw.get("allow_cron", True)),
    )


def _expand_mapping(
    raw: Any,
    env_values: Mapping[str, str] | None = None,
) -> dict[str, str]:
    if not isinstance(raw, dict):
        raise ValueError("MCP headers/env 必须是对象")
    return {
        str(key): _expand_env_refs(str(value), env_values)
        for key, value in raw.items()
    }


def _expand_env_refs(
    value: str,
    env_values: Mapping[str, str] | None = None,
) -> str:
    merged_env = dict(os.environ)
    merged_env.update(env_values or {})
    for name, env_value in merged_env.items():
        value = value.replace("${" + name + "}", env_value)
    return value


def _normalize_dotenv_values(raw: Mapping[str, Any]) -> dict[str, str]:
    return {
        str(key): str(value)
        for key, value in raw.items()
        if value is not None
    }
