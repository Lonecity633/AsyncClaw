"""LLM configuration loaded from dotenv and environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from AsyncClaw.config.providers import get_provider


@dataclass(frozen=True)
class LLMConfig:
    """OpenAI 兼容聊天补全 API 的运行时设置。"""

    api_key: str
    model: str
    base_url: str | None = None
    agent_max_steps: int = 8
    provider: str = "openai"


def load_llm_config(env_file: str | Path = ".env", override: bool = False) -> LLMConfig:
    """从 `.env` 和进程环境变量加载 LLM 配置。"""

    _load_dotenv(env_file, override=override)
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


def load_judge_llm_config(
    env_file: str | Path = ".env",
    override: bool = False,
) -> LLMConfig:
    """从独立的 `JUDGE_LLM_*` 环境变量加载 eval judge LLM 配置。"""

    _load_dotenv(env_file, override=override)
    provider = get_provider(os.getenv("JUDGE_LLM_PROVIDER") or "openai")

    return LLMConfig(
        provider=provider.name,
        api_key=_read_env("JUDGE_LLM_API_KEY") or "",
        base_url=_read_env("JUDGE_LLM_BASE_URL") or provider.base_url,
        model=_read_env("JUDGE_LLM_MODEL") or provider.default_model,
    )


def _load_dotenv(env_file: str | Path, *, override: bool) -> None:
    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "加载 .env 文件需要 python-dotenv。请使用 `pip install -e .` "
            "安装项目依赖。"
        ) from exc

    load_dotenv(dotenv_path=env_file, override=override)


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
