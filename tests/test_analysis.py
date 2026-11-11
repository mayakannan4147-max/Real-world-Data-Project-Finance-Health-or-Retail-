import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_retail import (  # noqa: E402
    customer_rfm,
    forecast_weekly_sales,
    load_transactions,
    weekly_features,
)


class AnalysisTests(unittest.TestCase):
    def test_load_transactions_excludes_cancelled_and_invalid_lines(self):
        source = pd.DataFrame(
            {
                "InvoiceNo": ["100", "C101", "102", "103"],
                "StockCode": ["A", "A", "A", "A"],
                "Quantity": [2, 1, 0, 1],
                "InvoiceDate": pd.to_datetime(
                    ["2011-01-01", "2011-01-02", "2011-01-03", None]
                ),
                "UnitPrice": [3.0, 3.0, 3.0, 3.0],
                "CustomerID": [1, 1, 1, 1],
                "Country": ["UK"] * 4,
            }
        )

        with patch("analyze_retail.pd.read_excel", return_value=source.copy()):
            cleaned, counts = load_transactions(Path("unused.xlsx"))

        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned["Revenue"].iloc[0], 6.0)
        self.assertEqual(counts["excluded_rows"], 3)

    def test_weekly_features_are_finite_and_index_aligned(self):
        dates = pd.Series(pd.date_range("2011-01-02", periods=3, freq="W-SUN"))
        features = weekly_features(dates, dates.iloc[0])

        self.assertEqual(list(features.columns), ["elapsed_years", "annual_sin", "annual_cos"])
        self.assertEqual(len(features), len(dates))
        self.assertTrue(features.notna().all().all())

    def test_forecast_splits_chronologically_and_returns_four_future_weeks(self):
        dates = pd.date_range("2011-01-02", periods=30, freq="W-SUN")
        sales = pd.Series(np.arange(30, dtype=float) * 10 + 100, index=dates)

        result = forecast_weekly_sales(sales)

        self.assertEqual(result["metrics"]["train_weeks"], 24)
        self.assertEqual(result["metrics"]["test_weeks"], 6)
        self.assertEqual(len(result["future"]), 4)
        self.assertEqual(result["actual"].index[0], dates[24])
        self.assertTrue(np.isfinite(result["metrics"]["model_mae"]))

    def test_customer_rfm_counts_distinct_invoices(self):
        transactions = pd.DataFrame(
            {
                "CustomerID": [1, 1, 1],
                "InvoiceNo": ["A", "A", "B"],
                "InvoiceDate": pd.to_datetime(
                    ["2011-01-01", "2011-01-01", "2011-01-10"]
                ),
                "Revenue": [2.0, 3.0, 4.0],
            }
        )

        result = customer_rfm(transactions)

        self.assertEqual(result.loc[0, "Frequency"], 2)
        self.assertEqual(result.loc[0, "Monetary"], 9.0)
        self.assertEqual(result.loc[0, "RecencyDays"], 1)


if __name__ == "__main__":
    unittest.main()
