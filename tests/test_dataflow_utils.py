"""Tests for dataflow utility functions."""

import unittest
import os
import tempfile
from datetime import datetime

import pandas as pd

from tradingagents.dataflows.utils import get_next_weekday, get_current_date, save_output


class TestGetNextWeekday(unittest.TestCase):
    def test_monday_unchanged(self):
        monday = datetime(2024, 1, 8)  # Monday
        result = get_next_weekday(monday)
        self.assertEqual(result, monday)

    def test_friday_unchanged(self):
        friday = datetime(2024, 1, 12)  # Friday
        result = get_next_weekday(friday)
        self.assertEqual(result, friday)

    def test_saturday_returns_next_monday(self):
        saturday = datetime(2024, 1, 13)  # Saturday
        result = get_next_weekday(saturday)
        expected_monday = datetime(2024, 1, 15)  # Next Monday
        self.assertEqual(result, expected_monday)

    def test_sunday_returns_next_monday(self):
        sunday = datetime(2024, 1, 14)  # Sunday
        result = get_next_weekday(sunday)
        expected_monday = datetime(2024, 1, 15)  # Next Monday
        self.assertEqual(result, expected_monday)

    def test_string_saturday_returns_next_monday(self):
        result = get_next_weekday("2024-01-13")  # Saturday string
        expected = datetime(2024, 1, 15)
        self.assertEqual(result, expected)

    def test_string_monday_unchanged(self):
        result = get_next_weekday("2024-01-08")  # Monday string
        expected = datetime(2024, 1, 8)
        self.assertEqual(result, expected)

    def test_result_is_datetime(self):
        result = get_next_weekday(datetime(2024, 1, 10))  # Wednesday
        self.assertIsInstance(result, datetime)


class TestGetCurrentDate(unittest.TestCase):
    def test_returns_string(self):
        result = get_current_date()
        self.assertIsInstance(result, str)

    def test_format_is_yyyy_mm_dd(self):
        result = get_current_date()
        parts = result.split("-")
        self.assertEqual(len(parts), 3)
        year, month, day = parts
        self.assertEqual(len(year), 4)
        self.assertEqual(len(month), 2)
        self.assertEqual(len(day), 2)

    def test_parseable_date(self):
        result = get_current_date()
        # Should not raise
        parsed = datetime.strptime(result, "%Y-%m-%d")
        self.assertIsNotNone(parsed)


class TestSaveOutput(unittest.TestCase):
    def test_saves_csv_when_path_provided(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            save_output(df, "test_tag", save_path=path)
            self.assertTrue(os.path.exists(path))
            loaded = pd.read_csv(path, index_col=0)
            self.assertEqual(list(loaded.columns), ["a", "b"])
        finally:
            os.unlink(path)

    def test_does_not_write_when_path_is_none(self):
        # Should not raise and nothing is written
        df = pd.DataFrame({"x": [1]})
        save_output(df, "tag", save_path=None)  # No exception expected

    def test_does_not_write_when_no_path_arg(self):
        df = pd.DataFrame({"x": [1]})
        save_output(df, "tag")  # Default save_path=None, no exception


if __name__ == "__main__":
    unittest.main()
