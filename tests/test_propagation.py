"""Tests for Propagator — initial state creation and graph args."""

import unittest

from tradingagents.graph.propagation import Propagator


class TestPropagatorCreateInitialState(unittest.TestCase):
    def setUp(self):
        self.propagator = Propagator()
        self.state = self.propagator.create_initial_state("AAPL", "2024-01-15")

    def test_state_has_messages(self):
        self.assertIn("messages", self.state)
        self.assertEqual(len(self.state["messages"]), 1)

    def test_messages_contain_company_name(self):
        role, content = self.state["messages"][0]
        self.assertEqual(role, "human")
        self.assertEqual(content, "AAPL")

    def test_company_of_interest_set(self):
        self.assertEqual(self.state["company_of_interest"], "AAPL")

    def test_trade_date_is_string(self):
        self.assertIsInstance(self.state["trade_date"], str)
        self.assertEqual(self.state["trade_date"], "2024-01-15")

    def test_trade_date_coerces_non_string(self):
        state = self.propagator.create_initial_state("TSLA", 20240115)
        self.assertEqual(state["trade_date"], "20240115")

    def test_investment_debate_state_initialized(self):
        ids = self.state["investment_debate_state"]
        self.assertEqual(ids["count"], 0)
        self.assertEqual(ids["bull_history"], "")
        self.assertEqual(ids["bear_history"], "")
        self.assertEqual(ids["history"], "")
        self.assertEqual(ids["current_response"], "")
        self.assertEqual(ids["judge_decision"], "")

    def test_risk_debate_state_initialized(self):
        rds = self.state["risk_debate_state"]
        self.assertEqual(rds["count"], 0)
        self.assertEqual(rds["latest_speaker"], "")
        self.assertEqual(rds["aggressive_history"], "")
        self.assertEqual(rds["conservative_history"], "")
        self.assertEqual(rds["neutral_history"], "")
        self.assertEqual(rds["history"], "")
        self.assertEqual(rds["judge_decision"], "")

    def test_report_fields_empty_initially(self):
        for field in ("market_report", "fundamentals_report", "sentiment_report", "news_report"):
            self.assertEqual(self.state[field], "", f"{field} should be empty initially")


class TestPropagatorGetGraphArgs(unittest.TestCase):
    def setUp(self):
        self.propagator = Propagator(max_recur_limit=50)

    def test_stream_mode_is_values(self):
        args = self.propagator.get_graph_args()
        self.assertEqual(args["stream_mode"], "values")

    def test_recursion_limit_set(self):
        args = self.propagator.get_graph_args()
        self.assertEqual(args["config"]["recursion_limit"], 50)

    def test_no_callbacks_by_default(self):
        args = self.propagator.get_graph_args()
        self.assertNotIn("callbacks", args["config"])

    def test_callbacks_included_when_provided(self):
        mock_callback = object()
        args = self.propagator.get_graph_args(callbacks=[mock_callback])
        self.assertIn("callbacks", args["config"])
        self.assertIn(mock_callback, args["config"]["callbacks"])

    def test_empty_callbacks_list_not_included(self):
        args = self.propagator.get_graph_args(callbacks=[])
        self.assertNotIn("callbacks", args["config"])

    def test_default_max_recur_limit(self):
        propagator = Propagator()
        args = propagator.get_graph_args()
        self.assertEqual(args["config"]["recursion_limit"], 100)


if __name__ == "__main__":
    unittest.main()
