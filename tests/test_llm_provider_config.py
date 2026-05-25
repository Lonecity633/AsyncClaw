from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from AsyncClaw.agent.llm import create_llm, create_openai_llm
from AsyncClaw.config import LLMConfig, load_llm_config
from AsyncClaw.providers import get_provider


class LLMProviderConfigTests(unittest.TestCase):
    def load_with_env(self, values: dict[str, str]) -> LLMConfig:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / ".env"
            env_file.write_text("", encoding="utf-8")
            with patch.dict(os.environ, values, clear=True):
                return load_llm_config(env_file=env_file)

    def test_default_provider_uses_openai_defaults(self) -> None:
        config = self.load_with_env({"OPENAI_API_KEY": "openai-key"})

        self.assertEqual(config.provider, "openai")
        self.assertEqual(config.api_key, "openai-key")
        self.assertEqual(config.model, "gpt-4o-mini")
        self.assertEqual(config.base_url, "https://api.openai.com/v1")

    def test_provider_specific_settings_take_precedence(self) -> None:
        config = self.load_with_env(
            {
                "LLM_PROVIDER": "deepseek",
                "LLM_API_KEY": "generic-key",
                "LLM_MODEL": "generic-model",
                "DEEPSEEK_API_KEY": "deepseek-key",
                "DEEPSEEK_MODEL": "deepseek-custom",
            }
        )

        self.assertEqual(config.provider, "deepseek")
        self.assertEqual(config.api_key, "deepseek-key")
        self.assertEqual(config.model, "deepseek-custom")
        self.assertEqual(config.base_url, "https://api.deepseek.com")

    def test_generic_base_url_overrides_provider_default(self) -> None:
        config = self.load_with_env(
            {
                "LLM_PROVIDER": "siliconflow",
                "SILICONFLOW_API_KEY": "siliconflow-key",
                "LLM_BASE_URL": "https://proxy.example/v1",
            }
        )

        self.assertEqual(config.provider, "siliconflow")
        self.assertEqual(config.api_key, "siliconflow-key")
        self.assertEqual(config.model, "deepseek-ai/DeepSeek-V3")
        self.assertEqual(config.base_url, "https://proxy.example/v1")

    def test_provider_base_url_takes_precedence_over_generic_base_url(self) -> None:
        config = self.load_with_env(
            {
                "LLM_PROVIDER": "deepseek",
                "DEEPSEEK_API_KEY": "deepseek-key",
                "DEEPSEEK_BASE_URL": "https://deepseek.example",
                "LLM_BASE_URL": "https://generic.example/v1",
            }
        )

        self.assertEqual(config.base_url, "https://deepseek.example")

    def test_xiaomi_aliases_are_supported(self) -> None:
        config = self.load_with_env(
            {
                "LLM_PROVIDER": "xiaomi",
                "XIAOMI_API_KEY": "xiaomi-key",
                "XIAOMI_MODEL": "xiaomi-model",
                "XIAOMI_BASE_URL": "https://xiaomi.example/v1",
            }
        )

        self.assertEqual(config.provider, "xiaomi")
        self.assertEqual(config.api_key, "xiaomi-key")
        self.assertEqual(config.model, "xiaomi-model")
        self.assertEqual(config.base_url, "https://xiaomi.example/v1")

    def test_unknown_provider_lists_supported_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "openai"):
            get_provider("missing")


class LLMFactoryProviderTests(unittest.TestCase):
    def test_factory_passes_config_to_openai_client(self) -> None:
        calls: list[dict[str, object]] = []

        class FakeOpenAI:
            def __init__(self, **kwargs):
                calls.append(kwargs)
                self.chat = types.SimpleNamespace(
                    completions=types.SimpleNamespace(create=lambda **_: {"ok": True})
                )

        fake_openai = types.SimpleNamespace(OpenAI=FakeOpenAI)
        config = LLMConfig(
            api_key="deepseek-key",
            model="deepseek-v4-flash",
            base_url="https://api.deepseek.com",
            provider="deepseek",
        )

        with patch.dict(sys.modules, {"openai": fake_openai}):
            llm = create_openai_llm(config)

        self.assertEqual(calls, [{"api_key": "deepseek-key", "base_url": "https://api.deepseek.com"}])
        self.assertEqual(llm.model, "deepseek-v4-flash")

    def test_create_llm_alias_uses_same_factory(self) -> None:
        calls: list[dict[str, object]] = []

        class FakeOpenAI:
            def __init__(self, **kwargs):
                calls.append(kwargs)
                self.chat = types.SimpleNamespace(
                    completions=types.SimpleNamespace(create=lambda **_: {"ok": True})
                )

        fake_openai = types.SimpleNamespace(OpenAI=FakeOpenAI)
        config = LLMConfig(
            api_key="key",
            model="model",
            base_url="https://example.test/v1",
            provider="openai",
        )

        with patch.dict(sys.modules, {"openai": fake_openai}):
            llm = create_llm(config)

        self.assertEqual(calls, [{"api_key": "key", "base_url": "https://example.test/v1"}])
        self.assertEqual(llm.model, "model")

    def test_factory_rejects_anthropic_without_compatible_base_url(self) -> None:
        config = LLMConfig(
            api_key="anthropic-key",
            model="claude-3-5-haiku-latest",
            base_url=None,
            provider="anthropic",
        )

        with self.assertRaisesRegex(ValueError, "OpenAI 兼容端点"):
            create_openai_llm(config)


if __name__ == "__main__":
    unittest.main()
