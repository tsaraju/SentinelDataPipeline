"""Load SENTINEL tables from Supabase, audit data quality, and verify features."""

from __future__ import annotations

import argparse
import bisect
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psycopg
from dotenv import load_dotenv

TABLES = (
    "customers",
    "merchants",
    "devices",
    "transactions",
    "payment_outcomes",
    "labels",
    "features",
)
PRIMARY_KEYS = {
    "customers": "customer_id",
    "merchants": "merchant_id",
    "devices": "device_id",
    "transactions": "transaction_id",
    "payment_outcomes": "transaction_id",
    "labels": "transaction_id",
    "features": "transaction_id",
}
SPLITS = ("warmup", "train", "validation", "holdout")
FEATURE_COLUMNS = (
    "amount_to_avg_ratio",
    "is_new_device",
    "customer_merchant_distance_km",
    "is_night",
    "txn_count_last_1h",
    "txn_amount_last_24h_paise",
)
EXPECTED_SQL_TYPES = {
    "customers": {
        "customer_id": "text",
        "created_at": "timestamp with time zone",
        "customer_age": "integer",
        "home_city": "text",
        "region": "text",
        "is_synthetic": "boolean",
    },
    "merchants": {
        "merchant_id": "text",
        "synthetic_name": "text",
        "merchant_category": "text",
        "merchant_location": "jsonb",
        "is_synthetic": "boolean",
    },
    "devices": {
        "device_id": "text",
        "device_kind": "text",
        "is_synthetic": "boolean",
    },
    "transactions": {
        "transaction_id": "text",
        "customer_id": "text",
        "merchant_id": "text",
        "device_id": "text",
        "amount_paise": "bigint",
        "transaction_amount": "numeric",
        "payment_channel": "text",
        "merchant_category": "text",
        "transaction_time": "timestamp with time zone",
        "customer_age": "integer",
        "customer_location": "jsonb",
        "merchant_location": "jsonb",
        "account_age": "integer",
        "previous_transactions": "integer",
        "average_transaction_value": "numeric",
        "failed_transactions": "integer",
        "split": "text",
        "is_synthetic": "boolean",
    },
    "payment_outcomes": {
        "transaction_id": "text",
        "outcome_time": "timestamp with time zone",
        "status": "text",
        "failure_code": "text",
        "is_synthetic": "boolean",
    },
    "labels": {
        "transaction_id": "text",
        "fraud_label": "smallint",
        "label_source": "text",
        "label_available_at": "timestamp with time zone",
        "split": "text",
        "eligible_at_next_split_start": "boolean",
        "is_synthetic": "boolean",
    },
    "features": {
        "transaction_id": "text",
        "feature_as_of": "timestamp with time zone",
        "amount_to_avg_ratio": "double precision",
        "is_new_device": "boolean",
        "customer_merchant_distance_km": "double precision",
        "is_new_city": "boolean",
        "hour_of_day": "integer",
        "is_night": "boolean",
        "has_recent_failures": "boolean",
        "is_new_account": "boolean",
        "txn_count_last_1h": "integer",
        "txn_amount_last_24h_paise": "bigint",
        "distinct_customers_per_device_7d": "integer",
        "combined_risk_flags": "integer",
        "mcc_target_encoded_rate": "double precision",
        "mcc_encoding_as_of": "timestamp with time zone",
        "mcc_encoding_count": "integer",
        "missing_customer_location": "boolean",
        "missing_device": "boolean",
        "cold_start": "boolean",
        "reason_codes": "jsonb",
        "feature_version": "text",
    },
}
REASON_CODE_NAMES = {
    "R01_NEW_DEVICE": "New device",
    "R02_GEO_MISMATCH": "Geo mismatch",
    "R03_ODD_HOUR": "Odd hour",
    "R04_FAILED_ATTEMPTS": "Failed attempts",
    "R05_AMOUNT_SPIKE": "Amount spike",
    "R06_NEW_ACCOUNT": "New account",
    "R07_VELOCITY": "Velocity",
    "R08_HIGH_RISK_MCC": "High-risk category",
}
OUTPUT_DIR = Path("reports")
SCREENSHOT_DIR = Path("screenshots")


def load_tables(
    database_url: str,
) -> tuple[dict[str, pd.DataFrame], dict[tuple[str, str], str]]:
    """Fetch each required public table into its own pandas DataFrame."""
    dataframes: dict[str, pd.DataFrame] = {}
    with psycopg.connect(database_url, sslmode="require") as connection:
        for table in TABLES:
            with connection.cursor() as cursor:
                cursor.execute(f"SELECT * FROM public.{table}")
                columns = [column.name for column in cursor.description]
                dataframes[table] = pd.DataFrame.from_records(
                    cursor.fetchall(), columns=columns
                )
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT table_name, column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = ANY(%s)
                """,
                (list(TABLES),),
            )
            database_types = {
                (table, column): sql_type
                for table, column, sql_type in cursor.fetchall()
            }
    return dataframes, database_types


def _json_safe_duplicate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    canonical = frame.copy()
    for column in canonical.columns:
        if canonical[column].dtype == object:
            canonical[column] = canonical[column].map(
                lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"))
                if isinstance(value, (dict, list))
                else value
            )
    return canonical


def audit_data(
    dataframes: dict[str, pd.DataFrame],
    database_types: dict[tuple[str, str], str],
) -> tuple[dict[str, Any], pd.DataFrame]:
    transactions = dataframes["transactions"]
    amount = transactions["transaction_amount"].astype(float)
    q1, q3 = amount.quantile([0.25, 0.75])
    iqr_fence = q3 + 1.5 * (q3 - q1)
    nulls = {
        table: {
            column: int(count)
            for column, count in frame.isna().sum().items()
            if count
        }
        for table, frame in dataframes.items()
    }
    duplicates = {
        table: {
            "exact_rows": int(_json_safe_duplicate_frame(frame).duplicated().sum()),
            "primary_key": PRIMARY_KEYS[table],
            "duplicate_keys": int(frame[PRIMARY_KEYS[table]].duplicated().sum()),
        }
        for table, frame in dataframes.items()
    }
    type_mismatches = [
        {
            "table": table,
            "column": column,
            "expected": expected,
            "actual": database_types.get((table, column), "missing"),
        }
        for table, columns in EXPECTED_SQL_TYPES.items()
        for column, expected in columns.items()
        if database_types.get((table, column)) != expected
    ]
    split_counts = (
        transactions["split"]
        .value_counts()
        .reindex(SPLITS, fill_value=0)
        .astype(int)
        .to_dict()
    )
    report = {
        "rows": {table: len(frame) for table, frame in dataframes.items()},
        "nulls_by_column": nulls,
        "duplicates": duplicates,
        "wrong_types": len(type_mismatches),
        "type_mismatches": type_mismatches,
        "database_columns_checked": len(database_types),
        "amount": {
            "min": float(amount.min()),
            "q1": float(q1),
            "median": float(amount.median()),
            "q3": float(q3),
            "iqr_upper_fence": float(iqr_fence),
            "above_iqr_fence": int((amount > iqr_fence).sum()),
            "above_10_lakh": int((amount > 1_000_000).sum()),
            "max": float(amount.max()),
            "nonpositive": int((amount <= 0).sum()),
            "paise_mismatches": int(
                (
                    transactions["amount_paise"].astype("int64")
                    != (amount * 100).round().astype("int64")
                ).sum()
            ),
        },
        "split_counts": split_counts,
    }
    null_rows = [
        {"table": table, "column": column, "null_count": count}
        for table, columns in nulls.items()
        for column, count in columns.items()
    ]
    return report, pd.DataFrame(
        null_rows, columns=["table", "column", "null_count"]
    )


def _coordinate(location: Any, keys: tuple[str, ...]) -> float | None:
    if not isinstance(location, dict):
        return None
    for key in keys:
        if location.get(key) is not None:
            return float(location[key])
    return None


def _haversine_km(
    customer_location: Any, merchant_location: Any
) -> float | None:
    customer = (
        _coordinate(customer_location, ("lat", "latitude")),
        _coordinate(customer_location, ("lon", "lng", "longitude")),
    )
    merchant = (
        _coordinate(merchant_location, ("lat", "latitude")),
        _coordinate(merchant_location, ("lon", "lng", "longitude")),
    )
    if None in customer + merchant:
        return None
    lat1, lon1, lat2, lon2 = map(
        math.radians,
        (customer[0], customer[1], merchant[0], merchant[1]),
    )
    haversine = (
        math.sin((lat2 - lat1) / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin((lon2 - lon1) / 2) ** 2
    )
    return 6371 * 2 * math.asin(math.sqrt(min(1.0, haversine)))


def recompute_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Recompute requested features using only strictly earlier transactions."""
    records: dict[str, dict[str, Any]] = {}
    rows = transactions[
        [
            "transaction_id",
            "customer_id",
            "device_id",
            "amount_paise",
            "customer_location",
            "merchant_location",
            "transaction_time",
        ]
    ].copy()
    rows["_time"] = pd.to_datetime(rows["transaction_time"], utc=True)

    for _, group in rows.groupby("customer_id", sort=False):
        group = group.sort_values("_time", kind="mergesort")
        times = group["_time"].tolist()
        amounts = group["amount_paise"].astype("int64").to_numpy()
        prefix_amounts = np.concatenate(
            (np.array([0], dtype=np.int64), np.cumsum(amounts, dtype=np.int64))
        )
        devices = group["device_id"].tolist()
        device_times: dict[str, list[pd.Timestamp]] = defaultdict(list)
        for device, timestamp in zip(devices, times):
            if isinstance(device, str):
                device_times[device].append(timestamp)

        ids = group["transaction_id"].tolist()
        customer_locations = group["customer_location"].tolist()
        merchant_locations = group["merchant_location"].tolist()
        for index, timestamp in enumerate(times):
            prior = bisect.bisect_left(times, timestamp)
            start_90d = bisect.bisect_left(
                times, timestamp - pd.Timedelta(days=90)
            )
            start_1h = bisect.bisect_left(
                times, timestamp - pd.Timedelta(hours=1)
            )
            start_24h = bisect.bisect_left(
                times, timestamp - pd.Timedelta(hours=24)
            )
            transaction_id = ids[index]
            average = (
                (prefix_amounts[prior] - prefix_amounts[start_90d])
                / (prior - start_90d)
                if prior > start_90d
                else None
            )
            device = devices[index]
            is_new_device = None
            if isinstance(device, str):
                history = device_times[device]
                before_current = bisect.bisect_left(history, timestamp)
                before_90d = bisect.bisect_left(
                    history, timestamp - pd.Timedelta(days=90)
                )
                is_new_device = before_current == before_90d

            local_time = timestamp.tz_convert("Asia/Kolkata")
            records[transaction_id] = {
                "amount_to_avg_ratio": (
                    round(int(amounts[index]) / average, 4)
                    if average is not None
                    else None
                ),
                "is_new_device": is_new_device,
                "customer_merchant_distance_km": (
                    round(distance, 2)
                    if (
                        distance := _haversine_km(
                            customer_locations[index], merchant_locations[index]
                        )
                    )
                    is not None
                    else None
                ),
                "is_night": local_time.hour < 5,
                "txn_count_last_1h": prior - start_1h,
                "txn_amount_last_24h_paise": int(
                    prefix_amounts[prior] - prefix_amounts[start_24h]
                ),
            }
    return pd.DataFrame.from_dict(records, orient="index").rename_axis(
        "transaction_id"
    )


def compare_features(
    calculated: pd.DataFrame, reference: pd.DataFrame, transactions: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare calculated values to stored values overall and per split."""
    expected = reference.set_index("transaction_id")
    split_by_id = transactions.set_index("transaction_id")["split"]
    result: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    for feature in FEATURE_COLUMNS:
        left = calculated[feature]
        right = expected[feature].reindex(calculated.index)
        if feature in ("amount_to_avg_ratio", "customer_merchant_distance_km"):
            decimals = 4 if feature == "amount_to_avg_ratio" else 2
            left = pd.to_numeric(left, errors="coerce").round(decimals)
            right = pd.to_numeric(right, errors="coerce").round(decimals)
        equal = (left.isna() & right.isna()) | (
            left.eq(right) & left.notna() & right.notna()
        )
        for split in ("all", *SPLITS):
            selected = (
                pd.Series(True, index=equal.index)
                if split == "all"
                else split_by_id.reindex(equal.index).eq(split)
            )
            total = int(selected.sum())
            matches = int((equal & selected).sum())
            result.append(
                {
                    "feature": feature,
                    "split": split,
                    "matches": matches,
                    "total": total,
                    "match_rate_pct": round(matches * 100 / total, 4)
                    if total
                    else None,
                }
            )
        for transaction_id in equal.index[~equal]:
            mismatches.append(
                {
                    "transaction_id": transaction_id,
                    "feature": feature,
                    "calculated": left.loc[transaction_id],
                    "reference": right.loc[transaction_id],
                    "split": split_by_id.loc[transaction_id],
                }
            )
    return pd.DataFrame(result), pd.DataFrame(
        mismatches,
        columns=["transaction_id", "feature", "calculated", "reference", "split"],
    )


def compare_reason_codes(
    transactions: pd.DataFrame, features: pd.DataFrame
) -> pd.DataFrame:
    """Check R01-R08 assignments against the class-provided reason-code list."""
    frame = transactions.set_index("transaction_id").join(
        features.set_index("transaction_id"), how="inner"
    )
    code_map = {
        transaction_id: set(value) if isinstance(value, list) else set()
        for transaction_id, value in frame["reason_codes"].items()
    }
    conditions = {
        "R01_NEW_DEVICE": frame["is_new_device"].eq(True),
        "R02_GEO_MISMATCH": frame["customer_merchant_distance_km"].gt(500).fillna(
            False
        )
        | frame["is_new_city"].eq(True),
        "R03_ODD_HOUR": frame["is_night"].eq(True),
        "R04_FAILED_ATTEMPTS": frame["has_recent_failures"].eq(True),
        "R05_AMOUNT_SPIKE": frame["amount_to_avg_ratio"].ge(5).fillna(False),
        "R06_NEW_ACCOUNT": frame["account_age"].lt(30),
        "R07_VELOCITY": frame["txn_count_last_1h"].ge(4),
        "R08_HIGH_RISK_MCC": frame["merchant_category"].isin(
            ["electronics", "gaming", "gift_cards"]
        ),
    }
    rows = []
    for code, condition in conditions.items():
        assigned = condition.fillna(False)
        reference = pd.Series(
            [code in code_map[transaction_id] for transaction_id in frame.index],
            index=frame.index,
        )
        rows.append(
            {
                "code": code,
                "rule": REASON_CODE_NAMES[code],
                "assigned": int(assigned.sum()),
                "reference": int(reference.sum()),
                "disagreements": int((assigned != reference).sum()),
            }
        )
    return pd.DataFrame(rows)


def _save_screenshot(
    report: dict[str, Any], feature_matches: pd.DataFrame, path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    overall = feature_matches[feature_matches["split"] == "all"]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    figure.suptitle("Supabase Data Quality & Feature Validation", fontsize=16)
    axes[0].bar(
        list(report["split_counts"]),
        list(report["split_counts"].values()),
        color="#3478b8",
    )
    axes[0].set_title("Transactions per split")
    axes[0].set_ylabel("Rows")
    axes[0].ticklabel_format(axis="y", style="plain")
    axes[1].barh(overall["feature"], overall["match_rate_pct"], color="#2a9d8f")
    axes[1].set_xlim(0, 100)
    axes[1].set_title("Recomputed feature match rate")
    axes[1].set_xlabel("Match rate (%)")
    for index, value in enumerate(overall["match_rate_pct"]):
        axes[1].text(value - 2, index, f"{value:.1f}%", va="center", ha="right")
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def run(output_dir: Path = OUTPUT_DIR, screenshot_dir: Path = SCREENSHOT_DIR) -> None:
    load_dotenv()
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise SystemExit(
            "DATABASE_URL is not set. Copy .env.example to .env and add your "
            "Supabase PostgreSQL connection string."
        )

    dataframes, database_types = load_tables(database_url)
    report, nulls = audit_data(dataframes, database_types)
    calculated = recompute_features(dataframes["transactions"])
    feature_matches, feature_mismatches = compare_features(
        calculated, dataframes["features"], dataframes["transactions"]
    )
    reason_matches = compare_reason_codes(
        dataframes["transactions"], dataframes["features"]
    )
    report["feature_matches"] = feature_matches.to_dict(orient="records")
    report["reason_code_matches"] = reason_matches.to_dict(orient="records")
    report["notes"] = [
        "Rows sharing a timestamp are excluded from one another's history.",
        "Missing values match only when both calculated and reference values are null.",
        "MCC rule uses the class bundle's synthetic merchant categories.",
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "data_quality.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    nulls.to_csv(output_dir / "nulls_by_column.csv", index=False)
    feature_matches.to_csv(output_dir / "feature_match_rates.csv", index=False)
    feature_mismatches.to_csv(output_dir / "feature_mismatches.csv", index=False)
    reason_matches.to_csv(output_dir / "reason_code_match_rates.csv", index=False)
    screenshot_path = screenshot_dir / "data_quality_report.png"
    _save_screenshot(report, feature_matches, screenshot_path)

    print("Loaded pandas DataFrames:")
    for table, frame in dataframes.items():
        print(f"  {table}_df: {len(frame):,} rows x {len(frame.columns)} columns")
    print("\nTransaction rows per split:")
    for split, count in report["split_counts"].items():
        print(f"  {split:10} {count:>8,}")
    print("\nFeature match rates:")
    for row in feature_matches[feature_matches["split"] == "all"].itertuples():
        print(f"  {row.feature:38} {row.matches:>7,}/{row.total:<7,} {row.match_rate_pct:6.2f}%")
    print(f"\nFeature mismatches: {len(feature_mismatches):,}")
    print(f"Screenshot: {screenshot_path}")
    print(f"Reports: {output_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--screenshot-dir", type=Path, default=SCREENSHOT_DIR)
    args = parser.parse_args()
    run(args.output_dir, args.screenshot_dir)


if __name__ == "__main__":
    main()
