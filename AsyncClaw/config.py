"""从环境变量加载配置。"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from AsyncClaw.providers import get_provider


@dataclass(frozen=True)
class LLMConfig:
    """OpenAI 兼容聊天补全 API 的运行时设置。"""

    api_key: str
    model: str
    base_url: str | None = None
    agent_max_steps: int = 8
    provider: str = "openai"


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


def load_llm_config(env_file: str | Path = ".env", override: bool = False) -> LLMConfig:
    """从 `.env` 和进程环境变量加载 LLM 配置。"""

    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "加载 .env 文件需要 python-dotenv。请使用 `pip install -e .` "
            "安装项目依赖。"
        ) from exc

    load_dotenv(dotenv_path=env_file, override=override)
    max_steps = _read_int("AGENT_MAX_STEPS", default=8)
    provider = get_provider(os.getenv("LLM_PROVIDER"))

    return LLMConfig(
        provider=provider.name,
        api_key=_read_env(
            provider.api_key_env,
            *provider.api_key_env_aliases,
            "LLM_API_KEY",
        )
        or "",
        base_url=_read_env(
            provider.base_url_env,
            *provider.base_url_env_aliases,
            "LLM_BASE_URL",
        )
        or provider.base_url,
        model=_read_env(
            provider.model_env,
            *provider.model_env_aliases,
            "LLM_MODEL",
        )
        or provider.default_model,
        agent_max_steps=max_steps,
    )


def load_mcp_config(env_file: str | Path = ".env", override: bool = False) -> MCPConfig:
    """从 `.env` 指向的 JSON 文件加载 MCP 工具服务配置。"""

    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "加载 .env 文件需要 python-dotenv。请使用 `pip install -e .` "
            "安装项目依赖。"
        ) from exc

    env_path = Path(env_file)
    load_dotenv(dotenv_path=env_path, override=override)
    config_path = os.getenv("MCP_CONFIG")
    if not config_path:
        return MCPConfig()

    path = Path(config_path)
    if not path.is_absolute():
        base_dir = env_path.parent if env_path.parent != Path("") else Path.cwd()
        path = base_dir / path

    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_servers = payload if isinstance(payload, list) else payload.get("servers", [])
    if not isinstance(raw_servers, list):
        raise ValueError("MCP_CONFIG 中的 servers 必须是数组")

    servers = tuple(_parse_mcp_server(index, item) for index, item in enumerate(raw_servers))
    return MCPConfig(servers=servers)


def _parse_mcp_server(index: int, raw: Any) -> MCPServerConfig:
    if not isinstance(raw, dict):
        raise ValueError(f"MCP server #{index + 1} 必须是对象")

    name = str(raw.get("name") or "").strip()
    if not name:
        raise ValueError(f"MCP server #{index + 1} 缺少 name")

    transport = str(raw.get("transport") or "streamable-http").strip()
    headers = _expand_mapping(raw.get("headers") or {})
    env = _expand_mapping(raw.get("env") or {})
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


def _expand_mapping(raw: Any) -> dict[str, str]:
    if not isinstance(raw, dict):
        raise ValueError("MCP headers/env 必须是对象")
    return {str(key): _expand_env_refs(str(value)) for key, value in raw.items()}


def _expand_env_refs(value: str) -> str:
    for name, env_value in os.environ.items():
        value = value.replace("${" + name + "}", env_value)
    return value


def _read_env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def _read_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value == "":
        return default
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} 必须是整数") from exc
