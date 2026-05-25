"""OpenAI 兼容聊天补全客户端的轻量适配器。"""

from __future__ import annotations

from typing import Any

from AsyncClaw.config import LLMConfig, load_llm_config


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
    """根据 `.env` 支持的配置创建 OpenAI SDK 客户端。"""

    config = config or load_llm_config()
    if not config.api_key:
        raise ValueError("必须设置 OPENAI_API_KEY")

    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "真实 API 调用需要 openai。请使用 `pip install -r requirements.txt` "
            "安装项目依赖。"
        ) from exc

    client_kwargs: dict[str, Any] = {"api_key": config.api_key}
    if config.base_url:
        client_kwargs["base_url"] = config.base_url

    return OpenAICompatibleLLM(OpenAI(**client_kwargs), model=config.model)
