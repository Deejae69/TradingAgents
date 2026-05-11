"""Tests for agent_utils: build_instrument_context and create_msg_delete."""

import unittest
from unittest.mock import MagicMock

from tradingagents.agents.utils.agent_utils import build_instrument_context, create_msg_delete


class TestBuildInstrumentContext(unittest.TestCase):
    def test_contains_ticker(self):
        ctx = build_instrument_context("AAPL")
        self.assertIn("AAPL", ctx)

    def test_contains_exchange_suffix_note(self):
        ctx = build_instrument_context("CNC.TO")
        self.assertIn("exchange suffix", ctx)

    def test_ticker_with_hk_suffix(self):
        ctx = build_instrument_context("0700.HK")
        self.assertIn("0700.HK", ctx)

    def test_ticker_with_london_suffix(self):
        ctx = build_instrument_context("BP.L")
        self.assertIn("BP.L", ctx)

    def test_ticker_with_tokyo_suffix(self):
        ctx = build_instrument_context("7203.T")
        self.assertIn("7203.T", ctx)

    def test_returns_string(self):
        ctx = build_instrument_context("SPY")
        self.assertIsInstance(ctx, str)

    def test_mentions_tool_call_instruction(self):
        ctx = build_instrument_context("MSFT")
        self.assertIn("tool call", ctx.lower())

    def test_plain_us_ticker(self):
        ctx = build_instrument_context("TSLA")
        self.assertIn("TSLA", ctx)


class TestCreateMsgDelete(unittest.TestCase):
    def _make_message(self, msg_id):
        msg = MagicMock()
        msg.id = msg_id
        return msg

    def test_returns_callable(self):
        delete_fn = create_msg_delete()
        self.assertTrue(callable(delete_fn))

    def test_removes_all_existing_messages(self):
        from langchain_core.messages import RemoveMessage

        delete_fn = create_msg_delete()
        messages = [self._make_message("id1"), self._make_message("id2")]
        state = {"messages": messages}
        result = delete_fn(state)

        removal_ops = [m for m in result["messages"] if isinstance(m, RemoveMessage)]
        self.assertEqual(len(removal_ops), 2)
        removed_ids = {m.id for m in removal_ops}
        self.assertIn("id1", removed_ids)
        self.assertIn("id2", removed_ids)

    def test_adds_placeholder_human_message(self):
        from langchain_core.messages import HumanMessage

        delete_fn = create_msg_delete()
        state = {"messages": [self._make_message("id1")]}
        result = delete_fn(state)

        human_messages = [m for m in result["messages"] if isinstance(m, HumanMessage)]
        self.assertEqual(len(human_messages), 1)
        self.assertEqual(human_messages[0].content, "Continue")

    def test_empty_state_adds_only_placeholder(self):
        from langchain_core.messages import HumanMessage, RemoveMessage

        delete_fn = create_msg_delete()
        state = {"messages": []}
        result = delete_fn(state)

        removal_ops = [m for m in result["messages"] if isinstance(m, RemoveMessage)]
        human_messages = [m for m in result["messages"] if isinstance(m, HumanMessage)]
        self.assertEqual(len(removal_ops), 0)
        self.assertEqual(len(human_messages), 1)

    def test_each_call_returns_independent_function(self):
        fn1 = create_msg_delete()
        fn2 = create_msg_delete()
        self.assertIsNot(fn1, fn2)


if __name__ == "__main__":
    unittest.main()
