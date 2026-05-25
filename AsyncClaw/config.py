"""从环境变量加载配置。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LLMConfig:
    """OpenAI 兼容聊天补全 API 的运行时设置。"""

    api_key: str
    model: str
    base_url: str | None = None
    agent_max_steps: int = 8


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

    return LLMConfig(
        api_key=os.getenv("OPENAI_API_KEY", ""),
        base_url=os.getenv("OPENAI_BASE_URL") or None,
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        agent_max_steps=max_steps,
    )


def _read_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value == "":
        return default
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} 必须是整数") from exc
