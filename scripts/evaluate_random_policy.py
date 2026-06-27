#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from deepblue_auv_rl.evaluation.evaluate import (
    ALL_STAGES,
    baseline_output_paths,
    evaluate_episodes,
    make_env,
    positive_int,
    print_summary,
    resolve_output_dir,
    save_baseline_summary_csv,
    save_evaluation_outputs,
    summarize_results,
    summary_row,
)


DEFAULT_EPISODES = 50
DEFAULT_SEED = 2000
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "eval" / "baselines" / "random_policy"


def random_policy(observation: Any, info: dict[str, Any], env: Any) -> int:
    del observation, info
    return int(env.action_space.sample())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a random-action baseline in AUVTargetEnv."
    )
    parser.add_argument(
        "--episodes",
        type=positive_int,
        default=DEFAULT_EPISODES,
        help="Number of episodes per stage.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Base seed. Episode i uses seed + i for every stage.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_DIR.relative_to(REPO_ROOT)),
        help="Directory where result files are written.",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=ALL_STAGES,
        default=list(ALL_STAGES),
        help="Stages to evaluate.",
    )
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    output_dir = resolve_output_dir(args.output, repo_root=REPO_ROOT)
    combined_rows: list[dict[str, Any]] = []

    for stage in args.stages:
        print(f"\nEvaluating random policy on stage: {stage}")
        env = make_env(stage)

        try:
            stage_start = time.time()
            results = evaluate_episodes(
                env=env,
                policy_fn=random_policy,
                episodes=args.episodes,
                seed=args.seed,
                stage=stage,
                verbose=True,
            )
            elapsed_time = time.time() - stage_start
        finally:
            env.close()

        summary = summarize_results(results)
        metadata = {
            "policy": "random",
            "stage": stage,
            "episodes": args.episodes,
            "seed": args.seed,
            "elapsed_time_seconds": elapsed_time,
        }
        results_json, episodes_csv, summary_json = baseline_output_paths(
            output_dir=output_dir,
            policy_name="random_policy",
            stage=stage,
            episodes=args.episodes,
            seed=args.seed,
        )

        save_evaluation_outputs(
            metadata=metadata,
            summary=summary,
            episode_rows=[result.to_row(include_stage=True) for result in results],
            results_json=results_json,
            episodes_csv=episodes_csv,
            summary_json=summary_json,
        )
        combined_rows.append(summary_row(summary, policy="random", stage=stage))
        print_summary(summary, title=f"Random Policy: {stage}")

    summary_csv = save_baseline_summary_csv(
        output_dir=output_dir,
        filename="random_policy_summary.csv",
        rows=combined_rows,
        prefix_columns=["policy", "stage"],
    )
    print(f"\nSaved random-policy baseline outputs to: {output_dir}")
    print(f"Combined summary CSV: {summary_csv}")


def main() -> None:
    try:
        run()
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from None


if __name__ == "__main__":
    main()
