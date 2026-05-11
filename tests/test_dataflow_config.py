"""Tests for the dataflows configuration module."""

import unittest

import tradingagents.dataflows.config as config_module
from tradingagents.default_config import DEFAULT_CONFIG


class TestDataflowConfig(unittest.TestCase):
    def setUp(self):
        # Reset config to a known state before each test
        config_module._config = None

    def tearDown(self):
        # Always restore config after each test
        config_module._config = None
        config_module.initialize_config()

    def test_initialize_config_populates_config(self):
        config_module._config = None
        config_module.initialize_config()
        self.assertIsNotNone(config_module._config)

    def test_initialize_config_idempotent(self):
        config_module.initialize_config()
        first = config_module.get_config()
        config_module.initialize_config()
        second = config_module.get_config()
        self.assertEqual(first, second)

    def test_get_config_returns_defaults(self):
        cfg = config_module.get_config()
        self.assertIn("llm_provider", cfg)
        self.assertEqual(cfg["llm_provider"], DEFAULT_CONFIG["llm_provider"])

    def test_get_config_returns_copy(self):
        cfg1 = config_module.get_config()
        cfg1["llm_provider"] = "mutated_value"
        cfg2 = config_module.get_config()
        self.assertNotEqual(cfg2["llm_provider"], "mutated_value")

    def test_set_config_updates_value(self):
        config_module.initialize_config()
        config_module.set_config({"llm_provider": "anthropic"})
        cfg = config_module.get_config()
        self.assertEqual(cfg["llm_provider"], "anthropic")

    def test_set_config_initializes_if_needed(self):
        config_module._config = None
        config_module.set_config({"llm_provider": "google"})
        cfg = config_module.get_config()
        self.assertEqual(cfg["llm_provider"], "google")

    def test_set_config_merges_not_replaces(self):
        config_module.initialize_config()
        original_results_dir = config_module.get_config()["results_dir"]
        config_module.set_config({"llm_provider": "anthropic"})
        cfg = config_module.get_config()
        # Unrelated keys should still be present
        self.assertEqual(cfg["results_dir"], original_results_dir)

    def test_get_config_triggers_initialize_if_not_done(self):
        config_module._config = None
        cfg = config_module.get_config()
        self.assertIsNotNone(cfg)
        self.assertIn("llm_provider", cfg)

    def test_config_has_data_vendors(self):
        cfg = config_module.get_config()
        self.assertIn("data_vendors", cfg)
        vendors = cfg["data_vendors"]
        self.assertIn("core_stock_apis", vendors)
        self.assertIn("technical_indicators", vendors)
        self.assertIn("fundamental_data", vendors)
        self.assertIn("news_data", vendors)


if __name__ == "__main__":
    unittest.main()
