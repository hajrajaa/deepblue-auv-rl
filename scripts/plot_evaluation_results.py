#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from deepblue_auv_rl.evaluation.evaluate import ALL_STAGES


DEFAULT_CROSS_SUMMARY = REPO_ROOT / "outputs" / "eval" / "cross_eval_summary.csv"
DEFAULT_BASELINE_SUMMARIES = (
    REPO_ROOT
    / "outputs"
    / "eval"
    / "baselines"
    / "random_policy"
    / "random_policy_summary.csv",
    REPO_ROOT
    / "outputs"
    / "eval"
    / "baselines"
    / "target_seeking"
    / "target_seeking_summary.csv",
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "plots"

STAGE_LABELS = {
    "random_start_no_obstacles": "Random start",
    "moving_no_obstacles": "Moving target",
    "fixed_obstacles": "Fixed obstacle",
    "moving_obstacles": "Moving obstacle",
}

POLICY_LABELS = {
    "random": "Random",
    "random_policy": "Random",
    "target_seeking": "Target-seeking",
}

COMPARISON_PLOTS = [
    (
        "success_rate",
        "Success Rate Comparison",
        "Success rate (%)",
        "success_rate_comparison",
        True,
    ),
    (
        "collision_rate",
        "Collision Rate Comparison",
        "Collision rate (%)",
        "collision_rate_comparison",
        True,
    ),
    (
        "mean_final_distance",
        "Mean Final Distance Comparison",
        "Mean final distance to target",
        "mean_final_distance_comparison",
        False,
    ),
    (
        "mean_return",
        "Mean Return Comparison",
        "Mean return",
        "mean_return_comparison",
        False,
    ),
    (
        "mean_episode_length",
        "Episode Length Comparison",
        "Mean episode length",
        "episode_length_comparison",
        False,
    ),
]

HEATMAP_PLOTS = [
    (
        "success_rate",
        "Cross-Evaluation Success Rate",
        "cross_eval_success_rate_heatmap",
        "Success rate (%)",
        "viridis",
    ),
    (
        "collision_rate",
        "Cross-Evaluation Collision Rate",
        "cross_eval_collision_rate_heatmap",
        "Collision rate (%)",
        "magma_r",
    ),
]


def resolve_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def parse_float(row: dict[str, str], column: str, *, source: Path) -> float:
    value = row.get(column)
    if value is None or value == "":
        raise ValueError(f"Missing column '{column}' in {source}")
    return float(value)


def stage_label(stage: str) -> str:
    return STAGE_LABELS.get(stage, stage.replace("_", " "))


def extract_stage_from_model(model_name: str) -> str | None:
    for stage in ALL_STAGES:
        if stage in model_name:
            return stage
    return None


def model_label(model_name: str) -> str:
    stage = extract_stage_from_model(model_name)
    if stage is not None:
        return f"PPO {stage_label(stage)}"

    stem = Path(model_name).stem
    return stem.replace("ppo_curriculum_", "PPO ").replace("_", " ")


def policy_label(policy_name: str) -> str:
    return POLICY_LABELS.get(policy_name, policy_name.replace("_", " ").title())


def load_cross_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Cross-evaluation summary CSV not found: {path}")

    rows: list[dict[str, Any]] = []
    for raw_row in read_csv_rows(path):
        trained_model = raw_row.get("trained_model")
        test_stage = raw_row.get("test_stage")
        if not trained_model or not test_stage:
            raise ValueError(
                f"{path} must contain 'trained_model' and 'test_stage' columns."
            )

        row: dict[str, Any] = {
            "series": model_label(trained_model),
            "trained_model": trained_model,
            "trained_stage": extract_stage_from_model(trained_model),
            "stage": test_stage,
        }
        for metric, *_ in COMPARISON_PLOTS:
            row[metric] = parse_float(raw_row, metric, source=path)
        rows.append(row)

    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def load_baseline_rows(paths: list[Path], stages: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            print(f"Skipping missing baseline summary: {path}")
            continue

        for raw_row in read_csv_rows(path):
            policy = raw_row.get("policy", path.stem)
            stage = raw_row.get("stage")
            if not stage or stage not in stages:
                continue

            row: dict[str, Any] = {
                "series": policy_label(policy),
                "trained_model": policy,
                "trained_stage": None,
                "stage": stage,
            }
            for metric, *_ in COMPARISON_PLOTS:
                row[metric] = parse_float(raw_row, metric, source=path)
            rows.append(row)

    return rows


def ordered_stages(rows: list[dict[str, Any]]) -> list[str]:
    present = {str(row["stage"]) for row in rows}
    ordered = [stage for stage in ALL_STAGES if stage in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def ordered_series(rows: list[dict[str, Any]]) -> list[str]:
    preferred_ppo: list[str] = []
    for stage in ALL_STAGES:
        label = f"PPO {stage_label(stage)}"
        if any(row["series"] == label for row in rows):
            preferred_ppo.append(label)

    baseline_order = [
        label
        for label in ("Random", "Target-seeking")
        if any(row["series"] == label for row in rows)
    ]
    known = set(preferred_ppo + baseline_order)
    rest = sorted({str(row["series"]) for row in rows} - known)
    return [*preferred_ppo, *baseline_order, *rest]


def metric_for_row(row: dict[str, Any], metric: str, *, percent: bool) -> float:
    value = float(row[metric])
    return 100.0 * value if percent else value


def save_figure(fig: Any, output_dir: Path, stem: str, formats: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for file_format in formats:
        output_path = output_dir / f"{stem}.{file_format}"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved {output_path}")


def plot_grouped_bar(
    *,
    rows: list[dict[str, Any]],
    metric: str,
    title: str,
    ylabel: str,
    output_stem: str,
    percent: bool,
    output_dir: Path,
    formats: list[str],
) -> None:
    stages = ordered_stages(rows)
    series = ordered_series(rows)
    values = {
        (str(row["stage"]), str(row["series"])): metric_for_row(
            row,
            metric,
            percent=percent,
        )
        for row in rows
    }

    x = np.arange(len(stages))
    bar_width = min(0.82 / max(len(series), 1), 0.16)
    offsets = (np.arange(len(series)) - (len(series) - 1) / 2.0) * bar_width

    fig_width = max(10.0, 1.8 * len(stages) + 0.8 * len(series))
    fig, ax = plt.subplots(figsize=(fig_width, 6.0))

    for series_idx, series_name in enumerate(series):
        series_values = [
            values.get((stage, series_name), math.nan)
            for stage in stages
        ]
        ax.bar(
            x + offsets[series_idx],
            series_values,
            width=bar_width,
            label=series_name,
        )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels([stage_label(stage) for stage in stages], rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)

    if percent:
        ax.set_ylim(bottom=0.0, top=100.0)

    fig.tight_layout()
    save_figure(fig, output_dir, output_stem, formats)
    plt.close(fig)


def ordered_heatmap_models(cross_rows: list[dict[str, Any]]) -> list[str]:
    model_by_stage: dict[str, str] = {}
    for row in cross_rows:
        trained_stage = row.get("trained_stage")
        if trained_stage and trained_stage not in model_by_stage:
            model_by_stage[str(trained_stage)] = str(row["trained_model"])

    ordered = [
        model_by_stage[stage]
        for stage in ALL_STAGES
        if stage in model_by_stage
    ]
    ordered.extend(
        sorted(
            {
                str(row["trained_model"])
                for row in cross_rows
            }
            - set(ordered)
        )
    )
    return ordered


def plot_heatmap(
    *,
    cross_rows: list[dict[str, Any]],
    metric: str,
    title: str,
    output_stem: str,
    colorbar_label: str,
    cmap_name: str,
    output_dir: Path,
    formats: list[str],
) -> None:
    stages = ordered_stages(cross_rows)
    trained_models = ordered_heatmap_models(cross_rows)
    value_lookup = {
        (str(row["trained_model"]), str(row["stage"])): 100.0 * float(row[metric])
        for row in cross_rows
    }

    matrix = np.full((len(trained_models), len(stages)), np.nan)
    for model_idx, trained_model in enumerate(trained_models):
        for stage_idx, stage in enumerate(stages):
            matrix[model_idx, stage_idx] = value_lookup.get(
                (trained_model, stage),
                np.nan,
            )

    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color="#eeeeee")

    fig_width = max(8.0, 1.5 * len(stages) + 4.0)
    fig_height = max(5.0, 0.7 * len(trained_models) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(matrix, cmap=cmap, vmin=0.0, vmax=100.0)

    ax.set_title(title)
    ax.set_xlabel("Test stage")
    ax.set_ylabel("Trained PPO model")
    ax.set_xticks(np.arange(len(stages)))
    ax.set_xticklabels([stage_label(stage) for stage in stages], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(trained_models)))
    ax.set_yticklabels([model_label(model) for model in trained_models])

    for model_idx in range(len(trained_models)):
        for stage_idx in range(len(stages)):
            value = matrix[model_idx, stage_idx]
            if not np.isnan(value):
                ax.text(
                    stage_idx,
                    model_idx,
                    f"{value:.0f}%",
                    ha="center",
                    va="center",
                    color="white" if value >= 50.0 else "black",
                    fontsize=9,
                )

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(colorbar_label)
    fig.tight_layout()
    save_figure(fig, output_dir, output_stem, formats)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create matplotlib plots from AUV RL evaluation summaries."
    )
    parser.add_argument(
        "--cross-summary",
        default=str(DEFAULT_CROSS_SUMMARY.relative_to(REPO_ROOT)),
        help="Combined PPO cross-evaluation CSV.",
    )
    parser.add_argument(
        "--baseline-summaries",
        nargs="*",
        default=[str(path.relative_to(REPO_ROOT)) for path in DEFAULT_BASELINE_SUMMARIES],
        help="Optional baseline summary CSV files to include in comparison plots.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_DIR.relative_to(REPO_ROOT)),
        help="Directory where plot files are saved.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "pdf", "svg"),
        default=["png", "pdf"],
        help="Figure formats to write.",
    )
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    cross_summary = resolve_path(args.cross_summary)
    baseline_summaries = [resolve_path(path) for path in args.baseline_summaries]
    output_dir = resolve_path(args.output)

    cross_rows = load_cross_rows(cross_summary)
    stages = ordered_stages(cross_rows)
    baseline_rows = load_baseline_rows(baseline_summaries, stages)
    comparison_rows = [*cross_rows, *baseline_rows]

    for metric, title, ylabel, output_stem, percent in COMPARISON_PLOTS:
        plot_grouped_bar(
            rows=comparison_rows,
            metric=metric,
            title=title,
            ylabel=ylabel,
            output_stem=output_stem,
            percent=percent,
            output_dir=output_dir,
            formats=args.formats,
        )

    for metric, title, output_stem, colorbar_label, cmap_name in HEATMAP_PLOTS:
        plot_heatmap(
            cross_rows=cross_rows,
            metric=metric,
            title=title,
            output_stem=output_stem,
            colorbar_label=colorbar_label,
            cmap_name=cmap_name,
            output_dir=output_dir,
            formats=args.formats,
        )

    print(f"\nCreated {len(COMPARISON_PLOTS) + len(HEATMAP_PLOTS)} figures in {output_dir}")


def main() -> None:
    try:
        run()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from None


if __name__ == "__main__":
    main()
