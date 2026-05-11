"""Tests for ConditionalLogic graph routing."""

import unittest
from unittest.mock import MagicMock

from tradingagents.graph.conditional_logic import ConditionalLogic


def _make_state_with_tool_calls():
    """Return an AgentState-like dict whose last message has tool_calls."""
    msg = MagicMock()
    msg.tool_calls = [{"name": "some_tool"}]
    return {"messages": [msg]}


def _make_state_without_tool_calls():
    """Return an AgentState-like dict whose last message has no tool_calls."""
    msg = MagicMock()
    msg.tool_calls = []
    return {"messages": [msg]}


def _make_debate_state(count, current_response):
    return {
        "messages": [],
        "investment_debate_state": {
            "count": count,
            "current_response": current_response,
            "bull_history": "",
            "bear_history": "",
        },
    }


def _make_risk_state(count, latest_speaker):
    return {
        "messages": [],
        "risk_debate_state": {
            "count": count,
            "latest_speaker": latest_speaker,
        },
    }


class TestShouldContinueAnalysts(unittest.TestCase):
    def setUp(self):
        self.logic = ConditionalLogic(max_debate_rounds=1, max_risk_discuss_rounds=1)

    # Market analyst routing
    def test_market_with_tool_calls_returns_tools_market(self):
        self.assertEqual(
            self.logic.should_continue_market(_make_state_with_tool_calls()),
            "tools_market",
        )

    def test_market_without_tool_calls_returns_msg_clear(self):
        self.assertEqual(
            self.logic.should_continue_market(_make_state_without_tool_calls()),
            "Msg Clear Market",
        )

    # Social analyst routing
    def test_social_with_tool_calls_returns_tools_social(self):
        self.assertEqual(
            self.logic.should_continue_social(_make_state_with_tool_calls()),
            "tools_social",
        )

    def test_social_without_tool_calls_returns_msg_clear(self):
        self.assertEqual(
            self.logic.should_continue_social(_make_state_without_tool_calls()),
            "Msg Clear Social",
        )

    # News analyst routing
    def test_news_with_tool_calls_returns_tools_news(self):
        self.assertEqual(
            self.logic.should_continue_news(_make_state_with_tool_calls()),
            "tools_news",
        )

    def test_news_without_tool_calls_returns_msg_clear(self):
        self.assertEqual(
            self.logic.should_continue_news(_make_state_without_tool_calls()),
            "Msg Clear News",
        )

    # Fundamentals analyst routing
    def test_fundamentals_with_tool_calls_returns_tools_fundamentals(self):
        self.assertEqual(
            self.logic.should_continue_fundamentals(_make_state_with_tool_calls()),
            "tools_fundamentals",
        )

    def test_fundamentals_without_tool_calls_returns_msg_clear(self):
        self.assertEqual(
            self.logic.should_continue_fundamentals(_make_state_without_tool_calls()),
            "Msg Clear Fundamentals",
        )


class TestShouldContinueDebate(unittest.TestCase):
    def test_count_reached_returns_research_manager(self):
        # max_debate_rounds=1, threshold is 2*1=2
        logic = ConditionalLogic(max_debate_rounds=1)
        state = _make_debate_state(count=2, current_response="Bull says buy")
        self.assertEqual(logic.should_continue_debate(state), "Research Manager")

    def test_count_exceeded_returns_research_manager(self):
        logic = ConditionalLogic(max_debate_rounds=1)
        state = _make_debate_state(count=10, current_response="Bull says buy")
        self.assertEqual(logic.should_continue_debate(state), "Research Manager")

    def test_bull_turn_routes_to_bear_researcher(self):
        logic = ConditionalLogic(max_debate_rounds=3)
        state = _make_debate_state(count=1, current_response="Bull Researcher response")
        self.assertEqual(logic.should_continue_debate(state), "Bear Researcher")

    def test_bear_turn_routes_to_bull_researcher(self):
        logic = ConditionalLogic(max_debate_rounds=3)
        state = _make_debate_state(count=1, current_response="Bear Researcher response")
        self.assertEqual(logic.should_continue_debate(state), "Bull Researcher")

    def test_higher_max_rounds_extends_debate(self):
        logic = ConditionalLogic(max_debate_rounds=3)
        # threshold = 2*3 = 6; count=5 should still debate
        state = _make_debate_state(count=5, current_response="Bear something")
        result = logic.should_continue_debate(state)
        self.assertNotEqual(result, "Research Manager")


class TestShouldContinueRiskAnalysis(unittest.TestCase):
    def test_count_reached_returns_portfolio_manager(self):
        # max_risk_discuss_rounds=1, threshold=3*1=3
        logic = ConditionalLogic(max_risk_discuss_rounds=1)
        state = _make_risk_state(count=3, latest_speaker="Aggressive Analyst")
        self.assertEqual(logic.should_continue_risk_analysis(state), "Portfolio Manager")

    def test_count_exceeded_returns_portfolio_manager(self):
        logic = ConditionalLogic(max_risk_discuss_rounds=1)
        state = _make_risk_state(count=99, latest_speaker="Neutral Analyst")
        self.assertEqual(logic.should_continue_risk_analysis(state), "Portfolio Manager")

    def test_aggressive_speaker_routes_to_conservative(self):
        logic = ConditionalLogic(max_risk_discuss_rounds=2)
        state = _make_risk_state(count=1, latest_speaker="Aggressive Analyst")
        self.assertEqual(logic.should_continue_risk_analysis(state), "Conservative Analyst")

    def test_conservative_speaker_routes_to_neutral(self):
        logic = ConditionalLogic(max_risk_discuss_rounds=2)
        state = _make_risk_state(count=1, latest_speaker="Conservative Analyst")
        self.assertEqual(logic.should_continue_risk_analysis(state), "Neutral Analyst")

    def test_neutral_or_other_speaker_routes_to_aggressive(self):
        logic = ConditionalLogic(max_risk_discuss_rounds=2)
        state = _make_risk_state(count=1, latest_speaker="Neutral Analyst")
        self.assertEqual(logic.should_continue_risk_analysis(state), "Aggressive Analyst")

    def test_initial_empty_speaker_routes_to_aggressive(self):
        logic = ConditionalLogic(max_risk_discuss_rounds=2)
        state = _make_risk_state(count=0, latest_speaker="")
        self.assertEqual(logic.should_continue_risk_analysis(state), "Aggressive Analyst")

    def test_higher_max_risk_rounds_extends_discussion(self):
        logic = ConditionalLogic(max_risk_discuss_rounds=3)
        # threshold = 3*3 = 9; count=8 should still discuss
        state = _make_risk_state(count=8, latest_speaker="Aggressive Analyst")
        result = logic.should_continue_risk_analysis(state)
        self.assertNotEqual(result, "Portfolio Manager")


if __name__ == "__main__":
    unittest.main()
