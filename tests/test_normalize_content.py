"""Tests for normalize_content helper in base_client."""

import unittest
from unittest.mock import MagicMock

from tradingagents.llm_clients.base_client import normalize_content


def _make_response(content):
    """Create a mock response object with the given content."""
    response = MagicMock()
    response.content = content
    return response


class TestNormalizeContentString(unittest.TestCase):
    def test_string_content_unchanged(self):
        response = _make_response("Hello, world!")
        result = normalize_content(response)
        self.assertEqual(result.content, "Hello, world!")

    def test_empty_string_unchanged(self):
        response = _make_response("")
        result = normalize_content(response)
        self.assertEqual(result.content, "")

    def test_returns_same_response_object(self):
        response = _make_response("text")
        result = normalize_content(response)
        self.assertIs(result, response)


class TestNormalizeContentList(unittest.TestCase):
    def test_single_text_block_extracted(self):
        response = _make_response([{"type": "text", "text": "Buy equities"}])
        result = normalize_content(response)
        self.assertEqual(result.content, "Buy equities")

    def test_multiple_text_blocks_joined(self):
        response = _make_response([
            {"type": "text", "text": "First paragraph"},
            {"type": "text", "text": "Second paragraph"},
        ])
        result = normalize_content(response)
        self.assertIn("First paragraph", result.content)
        self.assertIn("Second paragraph", result.content)

    def test_reasoning_block_discarded(self):
        response = _make_response([
            {"type": "reasoning", "reasoning": "internal thoughts"},
            {"type": "text", "text": "Actual answer"},
        ])
        result = normalize_content(response)
        self.assertEqual(result.content, "Actual answer")
        self.assertNotIn("internal thoughts", result.content)

    def test_mixed_string_and_dict_in_list(self):
        response = _make_response(["plain string", {"type": "text", "text": "dict text"}])
        result = normalize_content(response)
        self.assertIn("plain string", result.content)
        self.assertIn("dict text", result.content)

    def test_empty_list_produces_empty_string(self):
        response = _make_response([])
        result = normalize_content(response)
        self.assertEqual(result.content, "")

    def test_non_text_dicts_produce_empty(self):
        # Blocks with type != 'text' and no 'text' key should be ignored
        response = _make_response([{"type": "tool_use", "id": "xyz"}])
        result = normalize_content(response)
        self.assertEqual(result.content, "")

    def test_text_block_with_empty_text_skipped(self):
        response = _make_response([
            {"type": "text", "text": ""},
            {"type": "text", "text": "Non-empty"},
        ])
        result = normalize_content(response)
        self.assertEqual(result.content, "Non-empty")


if __name__ == "__main__":
    unittest.main()
