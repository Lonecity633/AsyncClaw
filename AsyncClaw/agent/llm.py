"""OpenAI 兼容聊天补全客户端的轻量适配器。"""

from __future__ import annotations

from typing import Any

from AsyncClaw.config import LLMConfig, load_llm_config
from AsyncClaw.providers import get_provider


class OpenAICompatibleLLM:
    """包装符合 OpenAI 聊天补全 API 形状的客户端。"""

    def __init__(self, client: Any, model: str):
        self.client = client
        self.model = model

    def create_chat_completion(self, **kwargs: Any) -> Any:
        """使用默认模型调用 `client.chat.completions.create`。"""

        kwargs.setdefault("model", self.model)
        return self.client.chat.completions.create(**kwargs)


def create_openai_llm(config: LLMConfig | None = None) -> OpenAICompatibleLLM:
    """根据 `.env` 支持的配置创建 OpenAI-compatible SDK 客户端。"""

    config = config or load_llm_config()
    provider = get_provider(config.provider)
    if not config.api_key:
        names = [provider.api_key_env, *provider.api_key_env_aliases, "LLM_API_KEY"]
        raise ValueError(f"必须设置 {provider.name} API Key，可用环境变量: {', '.join(names)}")

    base_url = config.base_url or provider.base_url
    if not provider.openai_compatible and not base_url:
        raise ValueError(
            f"{provider.name} 原生 API 暂未适配。请设置 LLM_BASE_URL 指向 OpenAI 兼容端点，"
            "或使用其他 OpenAI-compatible provider。"
        )

    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "真实 API 调用需要 openai。请使用 `pip install -r requirements.txt` "
            "安装项目依赖。"
        ) from exc

    client_kwargs: dict[str, Any] = {"api_key": config.api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    return OpenAICompatibleLLM(OpenAI(**client_kwargs), model=config.model)


def create_llm(config: LLMConfig | None = None) -> OpenAICompatibleLLM:
    """Create the configured OpenAI-compatible LLM client."""

    return create_openai_llm(config)
