"""Tests for the LLM client factory."""

import unittest

from tradingagents.llm_clients.factory import create_llm_client
from tradingagents.llm_clients.openai_client import OpenAIClient
from tradingagents.llm_clients.anthropic_client import AnthropicClient
from tradingagents.llm_clients.google_client import GoogleClient


class TestCreateLLMClient(unittest.TestCase):
    def test_openai_provider_returns_openai_client(self):
        client = create_llm_client("openai", "gpt-5-mini")
        self.assertIsInstance(client, OpenAIClient)

    def test_anthropic_provider_returns_anthropic_client(self):
        client = create_llm_client("anthropic", "claude-sonnet-4-6")
        self.assertIsInstance(client, AnthropicClient)

    def test_google_provider_returns_google_client(self):
        client = create_llm_client("google", "gemini-2.5-pro")
        self.assertIsInstance(client, GoogleClient)

    def test_xai_provider_returns_openai_client(self):
        # xAI uses the OpenAI-compatible API
        client = create_llm_client("xai", "grok-4-0709")
        self.assertIsInstance(client, OpenAIClient)

    def test_ollama_provider_returns_openai_client(self):
        client = create_llm_client("ollama", "qwen3:latest")
        self.assertIsInstance(client, OpenAIClient)

    def test_openrouter_provider_returns_openai_client(self):
        client = create_llm_client("openrouter", "org/some-model:free")
        self.assertIsInstance(client, OpenAIClient)

    def test_unsupported_provider_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            create_llm_client("unknown_provider", "some-model")
        self.assertIn("unknown_provider", str(ctx.exception))

    def test_provider_case_insensitive(self):
        client = create_llm_client("OpenAI", "gpt-5-mini")
        self.assertIsInstance(client, OpenAIClient)

    def test_base_url_passed_through(self):
        client = create_llm_client("openai", "gpt-5-mini", base_url="https://example.com/v1")
        self.assertEqual(client.base_url, "https://example.com/v1")

    def test_extra_kwargs_stored(self):
        client = create_llm_client("openai", "gpt-5-mini", timeout=30)
        self.assertEqual(client.kwargs.get("timeout"), 30)


class TestClientModelStorage(unittest.TestCase):
    def test_model_name_stored(self):
        client = create_llm_client("anthropic", "claude-haiku-4-5")
        self.assertEqual(client.model, "claude-haiku-4-5")

    def test_validate_model_valid(self):
        client = create_llm_client("openai", "gpt-5-mini")
        self.assertTrue(client.validate_model())

    def test_validate_model_invalid(self):
        client = create_llm_client("openai", "gpt-3-turbo-not-real")
        self.assertFalse(client.validate_model())


if __name__ == "__main__":
    unittest.main()
