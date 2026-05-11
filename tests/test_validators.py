"""Tests for LLM model name validators."""

import unittest

from tradingagents.llm_clients.validators import validate_model, VALID_MODELS


class TestValidateModelOpenAI(unittest.TestCase):
    def test_valid_openai_model(self):
        self.assertTrue(validate_model("openai", "gpt-5-mini"))

    def test_valid_openai_model_gpt5(self):
        self.assertTrue(validate_model("openai", "gpt-5"))

    def test_invalid_openai_model(self):
        self.assertFalse(validate_model("openai", "gpt-3-turbo-not-real"))

    def test_openai_case_insensitive_provider(self):
        self.assertTrue(validate_model("OpenAI", "gpt-5-mini"))


class TestValidateModelAnthropic(unittest.TestCase):
    def test_valid_anthropic_model(self):
        self.assertTrue(validate_model("anthropic", "claude-sonnet-4-6"))

    def test_valid_anthropic_opus(self):
        self.assertTrue(validate_model("anthropic", "claude-opus-4-5"))

    def test_invalid_anthropic_model(self):
        self.assertFalse(validate_model("anthropic", "claude-2-not-real"))

    def test_anthropic_case_insensitive_provider(self):
        self.assertTrue(validate_model("Anthropic", "claude-haiku-4-5"))


class TestValidateModelGoogle(unittest.TestCase):
    def test_valid_google_model(self):
        self.assertTrue(validate_model("google", "gemini-2.5-pro"))

    def test_valid_google_flash(self):
        self.assertTrue(validate_model("google", "gemini-2.5-flash"))

    def test_invalid_google_model(self):
        self.assertFalse(validate_model("google", "gemini-1-not-real"))

    def test_google_case_insensitive_provider(self):
        self.assertTrue(validate_model("Google", "gemini-2.5-pro"))


class TestValidateModelXAI(unittest.TestCase):
    def test_valid_xai_model(self):
        self.assertTrue(validate_model("xai", "grok-4-0709"))

    def test_invalid_xai_model(self):
        self.assertFalse(validate_model("xai", "grok-1-not-real"))


class TestValidateModelUnknownProvider(unittest.TestCase):
    def test_unknown_provider_returns_true(self):
        # Unknown providers are not validated — any model name is accepted
        self.assertTrue(validate_model("someunknownprovider", "any-model-name"))

    def test_ollama_accepts_any_model(self):
        self.assertTrue(validate_model("ollama", "custom-local-model:latest"))

    def test_openrouter_accepts_any_model(self):
        self.assertTrue(validate_model("openrouter", "org/some-model:free"))

    def test_ollama_case_insensitive(self):
        self.assertTrue(validate_model("Ollama", "some-model"))

    def test_openrouter_case_insensitive(self):
        self.assertTrue(validate_model("OpenRouter", "some-model"))


class TestValidModelConstants(unittest.TestCase):
    def test_valid_models_has_expected_providers(self):
        self.assertIn("openai", VALID_MODELS)
        self.assertIn("anthropic", VALID_MODELS)
        self.assertIn("google", VALID_MODELS)
        self.assertIn("xai", VALID_MODELS)

    def test_each_provider_has_at_least_one_model(self):
        for provider, models in VALID_MODELS.items():
            self.assertGreater(len(models), 0, f"{provider} should have at least one model")


if __name__ == "__main__":
    unittest.main()
