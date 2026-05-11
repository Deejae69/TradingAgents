"""Tests for Reflector — situation extraction and prompt generation."""

import unittest
from unittest.mock import MagicMock, patch

from tradingagents.graph.reflection import Reflector


def _make_state(
    market_report="Market Report",
    sentiment_report="Sentiment Report",
    news_report="News Report",
    fundamentals_report="Fundamentals Report",
    bull_history="Bull history",
    bear_history="Bear history",
    trader_plan="Trader plan",
    judge_decision_invest="Judge invest decision",
    judge_decision_risk="Risk judge decision",
):
    return {
        "market_report": market_report,
        "sentiment_report": sentiment_report,
        "news_report": news_report,
        "fundamentals_report": fundamentals_report,
        "investment_debate_state": {
            "bull_history": bull_history,
            "bear_history": bear_history,
            "judge_decision": judge_decision_invest,
        },
        "trader_investment_plan": trader_plan,
        "risk_debate_state": {
            "judge_decision": judge_decision_risk,
        },
    }


class TestReflectorPrompt(unittest.TestCase):
    def setUp(self):
        self.llm = MagicMock()
        self.reflector = Reflector(quick_thinking_llm=self.llm)

    def test_reflection_system_prompt_is_non_empty(self):
        prompt = self.reflector.reflection_system_prompt
        self.assertIsInstance(prompt, str)
        self.assertGreater(len(prompt), 50)

    def test_reflection_prompt_mentions_key_concepts(self):
        prompt = self.reflector.reflection_system_prompt
        self.assertIn("trading", prompt.lower())
        self.assertIn("decision", prompt.lower())


class TestReflectorExtractSituation(unittest.TestCase):
    def setUp(self):
        self.llm = MagicMock()
        self.reflector = Reflector(quick_thinking_llm=self.llm)

    def test_extract_situation_contains_all_reports(self):
        state = _make_state(
            market_report="MARKET_DATA",
            sentiment_report="SENTIMENT_DATA",
            news_report="NEWS_DATA",
            fundamentals_report="FUNDAMENTALS_DATA",
        )
        situation = self.reflector._extract_current_situation(state)
        self.assertIn("MARKET_DATA", situation)
        self.assertIn("SENTIMENT_DATA", situation)
        self.assertIn("NEWS_DATA", situation)
        self.assertIn("FUNDAMENTALS_DATA", situation)

    def test_extract_situation_handles_empty_reports(self):
        state = _make_state(
            market_report="",
            sentiment_report="",
            news_report="",
            fundamentals_report="",
        )
        # Should not raise
        situation = self.reflector._extract_current_situation(state)
        self.assertIsInstance(situation, str)


class TestReflectorReflectComponents(unittest.TestCase):
    def setUp(self):
        self.llm = MagicMock()
        self.llm.invoke.return_value = MagicMock(content="Reflection result")
        self.reflector = Reflector(quick_thinking_llm=self.llm)

    def test_reflect_bull_researcher_calls_add_situations(self):
        state = _make_state()
        memory = MagicMock()
        self.reflector.reflect_bull_researcher(state, 0.05, memory)
        memory.add_situations.assert_called_once()
        call_args = memory.add_situations.call_args[0][0]
        situation, result = call_args[0]
        self.assertIn("Market Report", situation)
        self.assertEqual(result, "Reflection result")

    def test_reflect_bear_researcher_calls_add_situations(self):
        state = _make_state()
        memory = MagicMock()
        self.reflector.reflect_bear_researcher(state, -0.03, memory)
        memory.add_situations.assert_called_once()

    def test_reflect_trader_calls_add_situations(self):
        state = _make_state()
        memory = MagicMock()
        self.reflector.reflect_trader(state, 0.02, memory)
        memory.add_situations.assert_called_once()

    def test_reflect_invest_judge_calls_add_situations(self):
        state = _make_state()
        memory = MagicMock()
        self.reflector.reflect_invest_judge(state, 0.01, memory)
        memory.add_situations.assert_called_once()

    def test_reflect_portfolio_manager_calls_add_situations(self):
        state = _make_state()
        memory = MagicMock()
        self.reflector.reflect_portfolio_manager(state, -0.01, memory)
        memory.add_situations.assert_called_once()

    def test_reflect_calls_llm_once_per_invocation(self):
        state = _make_state()
        memory = MagicMock()
        self.reflector.reflect_bull_researcher(state, 0.05, memory)
        self.assertEqual(self.llm.invoke.call_count, 1)

    def test_reflect_passes_returns_losses_in_message(self):
        state = _make_state()
        memory = MagicMock()
        self.reflector.reflect_trader(state, 0.08, memory)
        messages = self.llm.invoke.call_args[0][0]
        human_message = messages[1][1]
        self.assertIn("0.08", human_message)


if __name__ == "__main__":
    unittest.main()
