#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = REPO_ROOT / "outputs" / "eval" / "cross_eval"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "eval" / "cross_eval_summary.csv"

CSV_COLUMNS = [
    "trained_model",
    "test_stage",
    "success_rate",
    "collision_rate",
    "out_of_bounds_rate",
    "timeout_rate",
    "mean_return",
    "mean_final_distance",
    "mean_min_distance",
    "mean_episode_length",
]

STAGE_ORDER = {
    "fixed_no_obstacles": 0,
    "random_start_no_obstacles": 1,
    "moving_no_obstacles": 2,
    "fixed_obstacles": 3,
    "moving_obstacles": 4,
}


def resolve_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def load_summary_row(summary_path: Path) -> dict[str, Any]:
    with summary_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    metadata = payload.get("metadata", {})
    summary = payload.get("summary", {})

    model_path = metadata.get("model")
    trained_model = Path(model_path).name if model_path else summary_path.stem
    test_stage = metadata.get("stage")
    if not test_stage:
        raise ValueError(f"Missing metadata.stage in {summary_path}")

    row = {
        "trained_model": trained_model,
        "test_stage": test_stage,
    }

    for column in CSV_COLUMNS[2:]:
        if column not in summary:
            raise ValueError(f"Missing summary.{column} in {summary_path}")
        row[column] = summary[column]

    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a combined CSV from cross-evaluation summary JSON files."
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR.relative_to(REPO_ROOT)),
        help="Directory containing *_summary.json files from evaluate_model.py.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT.relative_to(REPO_ROOT)),
        help="CSV path to write.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = resolve_path(args.input_dir)
    output_path = resolve_path(args.output)

    summary_paths = sorted(input_dir.rglob("*_summary.json"))
    if not summary_paths:
        raise SystemExit(f"No *_summary.json files found in {input_dir}")

    rows = [load_summary_row(path) for path in summary_paths]
    rows.sort(
        key=lambda row: (
            row["trained_model"],
            STAGE_ORDER.get(str(row["test_stage"]), 99),
            row["test_stage"],
        )
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
