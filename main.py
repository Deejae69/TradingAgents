from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

# DEFAULT_CONFIG already applies TRADINGAGENTS_* env-var overrides
# (llm_provider, deep_think_llm, quick_think_llm, backend_url, etc.),
# so users can switch models or endpoints purely via .env without
# editing this script. Override individual keys here only when you
# want a hard-coded value that should ignore the environment.
config = DEFAULT_CONFIG.copy()

# Initialize with custom config
ta = TradingAgentsGraph(debug=True, config=config)

# --- Run 1: forward propagate ---
_, decision = ta.propagate("NVDA", "2024-05-10")
print("Decision:", decision)

# --- Self-learning: reflect on the outcome and update agent memories ---
# Pass the realized profit/loss from the trade (positive = profit, negative = loss).
# Each agent (Bull Researcher, Bear Researcher, Trader, Investment Judge,
# Portfolio Manager) will analyse its own reasoning against the actual outcome,
# extract actionable lessons, and store them in its BM25 memory so that future
# decisions on similar market conditions are informed by past experience.
#
# Uncomment and adjust the P&L value to enable the self-learning cycle:
# ta.reflect_and_remember(1000)  # e.g. position returned +$1 000
