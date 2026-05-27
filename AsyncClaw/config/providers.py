"""LLM provider defaults for OpenAI-compatible chat completion APIs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LLMProvider:
    """Configuration defaults for a model service provider."""

    name: str
    api_key_env: str
    model_env: str
    base_url_env: str
    default_model: str
    base_url: str | None = None
    openai_compatible: bool = True
    api_key_env_aliases: tuple[str, ...] = ()
    model_env_aliases: tuple[str, ...] = ()
    base_url_env_aliases: tuple[str, ...] = ()


SUPPORTED_PROVIDERS: dict[str, LLMProvider] = {
    "openai": LLMProvider(
        name="openai",
        api_key_env="OPENAI_API_KEY",
        model_env="OPENAI_MODEL",
        base_url_env="OPENAI_BASE_URL",
        default_model="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
    ),
    "deepseek": LLMProvider(
        name="deepseek",
        api_key_env="DEEPSEEK_API_KEY",
        model_env="DEEPSEEK_MODEL",
        base_url_env="DEEPSEEK_BASE_URL",
        default_model="deepseek-v4-flash",
        base_url="https://api.deepseek.com",
    ),
    "siliconflow": LLMProvider(
        name="siliconflow",
        api_key_env="SILICONFLOW_API_KEY",
        model_env="SILICONFLOW_MODEL",
        base_url_env="SILICONFLOW_BASE_URL",
        default_model="deepseek-ai/DeepSeek-V3",
        base_url="https://api.siliconflow.cn/v1",
    ),
    "xiaomi": LLMProvider(
        name="xiaomi",
        api_key_env="MIMO_API_KEY",
        model_env="MIMO_MODEL",
        base_url_env="MIMO_BASE_URL",
        default_model="mimo-v2.5-pro",
        base_url="https://api.xiaomimimo.com/v1",
        api_key_env_aliases=("XIAOMI_API_KEY",),
        model_env_aliases=("XIAOMI_MODEL",),
        base_url_env_aliases=("XIAOMI_BASE_URL",),
    ),
    "anthropic": LLMProvider(
        name="anthropic",
        api_key_env="ANTHROPIC_API_KEY",
        model_env="ANTHROPIC_MODEL",
        base_url_env="ANTHROPIC_BASE_URL",
        default_model="claude-3-5-haiku-latest",
        base_url=None,
        openai_compatible=False,
    ),
}


def get_provider(name: str | None) -> LLMProvider:
    """Return provider defaults by name."""

    normalized = (name or "openai").strip().lower()
    try:
        return SUPPORTED_PROVIDERS[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(SUPPORTED_PROVIDERS))
        raise ValueError(f"不支持的 LLM_PROVIDER: {name!r}。支持: {supported}") from exc
