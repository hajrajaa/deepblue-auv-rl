#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
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
DEFAULT_SEED = 3000
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "eval" / "baselines" / "target_seeking"

ACTION_FORWARD = 0
ACTION_TURN_LEFT = 1
ACTION_TURN_RIGHT = 2
ACTION_MOVE_UP = 3
ACTION_MOVE_DOWN = 4
ACTION_STOP = 5


def normalize_angle_deg(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def target_seeking_policy(observation: Any, info: dict[str, Any], env: Any) -> int:
    """Simple target-seeking controller with no obstacle avoidance."""
    target_dx = float(observation[0])
    target_dy = float(observation[1])
    target_dz = float(observation[2])
    distance = float(info.get("distance_to_target", observation[3]))

    config = getattr(env, "config", None)
    reach_threshold = float(getattr(config, "reach_threshold", 1.0))
    turn_degrees = float(getattr(config, "turn_degrees", 15.0))
    vertical_step = float(getattr(config, "vertical_step", 0.75))

    horizontal_distance = math.hypot(target_dx, target_dy)
    vertical_tolerance = max(0.35, 0.5 * vertical_step)

    if distance <= reach_threshold:
        return ACTION_STOP

    if horizontal_distance > 0.25:
        desired_yaw = math.degrees(math.atan2(target_dy, target_dx))
        current_yaw = float(getattr(env, "yaw_deg", 0.0))
        yaw_error = normalize_angle_deg(desired_yaw - current_yaw)
        turn_tolerance = max(5.0, 0.5 * turn_degrees)

        if yaw_error > turn_tolerance:
            return ACTION_TURN_LEFT
        if yaw_error < -turn_tolerance:
            return ACTION_TURN_RIGHT

    # Interleave depth corrections with forward motion so the AUV still makes
    # horizontal progress toward the target.
    current_step = int(getattr(env, "current_step", 0))
    should_adjust_depth = (
        abs(target_dz) > vertical_tolerance
        and (horizontal_distance <= 2.0 * reach_threshold or current_step % 3 == 2)
    )
    if should_adjust_depth:
        return ACTION_MOVE_UP if target_dz > 0.0 else ACTION_MOVE_DOWN

    if horizontal_distance > 0.5 * reach_threshold:
        return ACTION_FORWARD

    if abs(target_dz) > vertical_tolerance:
        return ACTION_MOVE_UP if target_dz > 0.0 else ACTION_MOVE_DOWN

    return ACTION_STOP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a simple target-seeking baseline in AUVTargetEnv."
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
        print(f"\nEvaluating target-seeking baseline on stage: {stage}")
        env = make_env(stage)

        try:
            stage_start = time.time()
            results = evaluate_episodes(
                env=env,
                policy_fn=target_seeking_policy,
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
            "policy": "target_seeking",
            "stage": stage,
            "episodes": args.episodes,
            "seed": args.seed,
            "elapsed_time_seconds": elapsed_time,
            "description": (
                "Heuristic controller that turns toward the target, moves forward, "
                "and intermittently corrects depth. It does not perform obstacle "
                "avoidance."
            ),
        }
        results_json, episodes_csv, summary_json = baseline_output_paths(
            output_dir=output_dir,
            policy_name="target_seeking",
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
        combined_rows.append(summary_row(summary, policy="target_seeking", stage=stage))
        print_summary(summary, title=f"Target-Seeking Baseline: {stage}")

    summary_csv = save_baseline_summary_csv(
        output_dir=output_dir,
        filename="target_seeking_summary.csv",
        rows=combined_rows,
        prefix_columns=["policy", "stage"],
    )
    print(f"\nSaved target-seeking baseline outputs to: {output_dir}")
    print(f"Combined summary CSV: {summary_csv}")


def main() -> None:
    try:
        run()
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from None


if __name__ == "__main__":
    main()
