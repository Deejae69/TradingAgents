"""Tests for FinancialSituationMemory (BM25-based memory)."""

import unittest

from tradingagents.agents.utils.memory import FinancialSituationMemory


class TestFinancialSituationMemoryEmpty(unittest.TestCase):
    def setUp(self):
        self.memory = FinancialSituationMemory("test")

    def test_get_memories_empty_returns_empty_list(self):
        result = self.memory.get_memories("any query")
        self.assertEqual(result, [])

    def test_get_memories_empty_with_n_greater_than_one(self):
        result = self.memory.get_memories("any query", n_matches=5)
        self.assertEqual(result, [])

    def test_bm25_is_none_initially(self):
        self.assertIsNone(self.memory.bm25)

    def test_documents_empty_initially(self):
        self.assertEqual(self.memory.documents, [])
        self.assertEqual(self.memory.recommendations, [])


class TestFinancialSituationMemoryAddAndRetrieve(unittest.TestCase):
    def setUp(self):
        self.memory = FinancialSituationMemory("test")

    def test_add_single_situation_and_retrieve(self):
        self.memory.add_situations([("Bull market with rising stocks", "Buy equities")])
        results = self.memory.get_memories("Bull market rising stocks")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["recommendation"], "Buy equities")
        self.assertEqual(results[0]["matched_situation"], "Bull market with rising stocks")

    def test_result_contains_required_keys(self):
        self.memory.add_situations([("Inflation rising", "Hedge with bonds")])
        results = self.memory.get_memories("Inflation")
        self.assertIn("matched_situation", results[0])
        self.assertIn("recommendation", results[0])
        self.assertIn("similarity_score", results[0])

    def test_similarity_score_is_a_number(self):
        self.memory.add_situations([("Tech sector correction", "Reduce tech exposure")])
        results = self.memory.get_memories("Tech correction")
        score = results[0]["similarity_score"]
        # BM25 scores are floats; exact range depends on corpus size
        self.assertIsInstance(float(score), float)

    def test_add_multiple_situations_returns_top_n(self):
        data = [
            ("Bull market rising stocks dividends", "Buy growth"),
            ("Inflation high interest rates bonds", "Hedge inflation"),
            ("Bear market recession unemployment", "Sell equities"),
        ]
        self.memory.add_situations(data)
        results = self.memory.get_memories("Bull market stocks", n_matches=2)
        self.assertEqual(len(results), 2)

    def test_most_relevant_result_ranked_first(self):
        data = [
            ("Tech sector correction selloff volatility", "Reduce tech"),
            ("High inflation rising interest rates bonds", "Buy bonds"),
        ]
        self.memory.add_situations(data)
        results = self.memory.get_memories("tech selloff volatility")
        # The tech-related situation should score higher
        self.assertEqual(results[0]["matched_situation"], "Tech sector correction selloff volatility")

    def test_n_matches_capped_at_document_count(self):
        self.memory.add_situations([("Single situation only", "Single recommendation")])
        results = self.memory.get_memories("single situation", n_matches=10)
        self.assertEqual(len(results), 1)

    def test_add_situations_incrementally(self):
        self.memory.add_situations([("Situation A", "Rec A")])
        self.memory.add_situations([("Situation B", "Rec B")])
        self.assertEqual(len(self.memory.documents), 2)
        results = self.memory.get_memories("Situation A B", n_matches=2)
        self.assertEqual(len(results), 2)


class TestFinancialSituationMemoryClear(unittest.TestCase):
    def test_clear_removes_all_documents(self):
        memory = FinancialSituationMemory("test")
        memory.add_situations([("Situation", "Recommendation")])
        memory.clear()
        self.assertEqual(memory.documents, [])
        self.assertEqual(memory.recommendations, [])
        self.assertIsNone(memory.bm25)

    def test_get_memories_after_clear_returns_empty(self):
        memory = FinancialSituationMemory("test")
        memory.add_situations([("Situation", "Recommendation")])
        memory.clear()
        result = memory.get_memories("Situation")
        self.assertEqual(result, [])


class TestFinancialSituationMemoryTokenize(unittest.TestCase):
    def setUp(self):
        self.memory = FinancialSituationMemory("test")

    def test_tokenize_lowercases(self):
        tokens = self.memory._tokenize("Bull MARKET Rising")
        self.assertIn("bull", tokens)
        self.assertIn("market", tokens)
        self.assertIn("rising", tokens)

    def test_tokenize_strips_punctuation(self):
        tokens = self.memory._tokenize("high-inflation! rates?")
        self.assertNotIn("high-inflation!", tokens)
        self.assertNotIn("rates?", tokens)

    def test_tokenize_empty_string(self):
        tokens = self.memory._tokenize("")
        self.assertEqual(tokens, [])

    def test_tokenize_alphanumeric_preserved(self):
        tokens = self.memory._tokenize("S&P500 ETF")
        self.assertIn("s", tokens)
        self.assertIn("p500", tokens)
        self.assertIn("etf", tokens)


class TestFinancialSituationMemoryName(unittest.TestCase):
    def test_name_stored(self):
        memory = FinancialSituationMemory("bull_memory")
        self.assertEqual(memory.name, "bull_memory")

    def test_config_param_ignored(self):
        memory = FinancialSituationMemory("test", config={"irrelevant": "value"})
        self.assertEqual(memory.name, "test")


if __name__ == "__main__":
    unittest.main()
