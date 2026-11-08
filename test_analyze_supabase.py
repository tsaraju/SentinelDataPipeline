import unittest

import pandas as pd

from analyze_supabase import recompute_features


class RecomputeFeaturesTests(unittest.TestCase):
    def test_histories_exclude_current_and_same_timestamp_transactions(self) -> None:
        transactions = pd.DataFrame(
            [
                {
                    "transaction_id": "first",
                    "customer_id": "customer",
                    "device_id": "device-a",
                    "amount_paise": 10_000,
                    "customer_location": {"lat": 28.6, "lon": 77.2},
                    "merchant_location": {"lat": 28.6, "lon": 77.2},
                    "transaction_time": "2025-01-01T00:00:00Z",
                },
                {
                    "transaction_id": "second",
                    "customer_id": "customer",
                    "device_id": "device-a",
                    "amount_paise": 20_000,
                    "customer_location": {"lat": 28.6, "lon": 77.2},
                    "merchant_location": {"lat": 28.6, "lon": 77.2},
                    "transaction_time": "2025-01-01T00:30:00Z",
                },
                {
                    "transaction_id": "same-time",
                    "customer_id": "customer",
                    "device_id": "device-b",
                    "amount_paise": 30_000,
                    "customer_location": {"lat": 28.6, "lon": 77.2},
                    "merchant_location": {"lat": 28.6, "lon": 77.2},
                    "transaction_time": "2025-01-01T00:30:00Z",
                },
            ]
        )

        result = recompute_features(transactions)

        self.assertTrue(pd.isna(result.loc["first", "amount_to_avg_ratio"]))
        self.assertTrue(result.loc["first", "is_new_device"])
        self.assertEqual(result.loc["second", "amount_to_avg_ratio"], 2.0)
        self.assertFalse(result.loc["second", "is_new_device"])
        self.assertTrue(result.loc["same-time", "is_new_device"])
        self.assertEqual(result.loc["second", "txn_count_last_1h"], 1)
        self.assertEqual(result.loc["same-time", "txn_count_last_1h"], 1)
        self.assertEqual(result.loc["second", "txn_amount_last_24h_paise"], 10_000)
        self.assertEqual(result.loc["same-time", "txn_amount_last_24h_paise"], 10_000)
        self.assertEqual(result.loc["second", "customer_merchant_distance_km"], 0)


if __name__ == "__main__":
    unittest.main()
