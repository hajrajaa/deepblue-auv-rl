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
    check_model_compatibility,
    evaluate_episodes,
    load_sb3_dependencies,
    make_env,
    make_sb3_policy,
    positive_int,
    print_summary,
    resolve_model_path,
    save_evaluation_outputs,
    str_to_bool,
    summarize_results,
)


MODEL_DIR = REPO_ROOT / "models"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "eval"


def resolve_output_paths(
    raw_output: str,
    *,
    model_path: Path,
    stage: str,
    episodes: int,
    seed: int,
    deterministic: bool,
) -> tuple[Path, Path, Path]:
    output_path = Path(raw_output).expanduser()
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path

    prediction_mode = "deterministic" if deterministic else "stochastic"
    output_stem = (
        f"eval_{model_path.stem}_{stage}_{episodes}eps_"
        f"seed_{seed}_{prediction_mode}"
    )

    if output_path.suffix.lower() == ".json":
        results_json = output_path.resolve()
        episodes_csv = results_json.with_name(f"{results_json.stem}_episodes.csv")
        summary_json = results_json.with_name(f"{results_json.stem}_summary.json")
    else:
        output_dir = output_path.resolve()
        results_json = output_dir / f"{output_stem}.json"
        episodes_csv = output_dir / f"{output_stem}_episodes.csv"
        summary_json = output_dir / f"{output_stem}_summary.json"

    results_json.parent.mkdir(parents=True, exist_ok=True)
    episodes_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    return results_json, episodes_csv, summary_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Stable-Baselines3 PPO model in AUVTargetEnv."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model checkpoint path. Bare names are resolved inside models/.",
    )
    parser.add_argument(
        "--stage",
        default="fixed_no_obstacles",
        choices=ALL_STAGES,
        help="Mission stage/environment configuration to evaluate.",
    )
    parser.add_argument(
        "--episodes",
        type=positive_int,
        default=10,
        help="Number of evaluation episodes.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed. Episode i uses seed + i.",
    )
    parser.add_argument(
        "--deterministic",
        nargs="?",
        const=True,
        default=False,
        type=str_to_bool,
        help="Use deterministic PPO actions. Can be passed as a flag or true/false.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_DIR.relative_to(REPO_ROOT)),
        help="Output directory, or a .json path for the combined results file.",
    )
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    model_path = resolve_model_path(
        args.model,
        repo_root=REPO_ROOT,
        model_dir=MODEL_DIR,
    )
    results_json, episodes_csv, summary_json = resolve_output_paths(
        args.output,
        model_path=model_path,
        stage=args.stage,
        episodes=args.episodes,
        seed=args.seed,
        deterministic=args.deterministic,
    )

    PPO, set_random_seed = load_sb3_dependencies()
    set_random_seed(args.seed)

    print("Loading PPO model:")
    print(f"  {model_path}")
    model = PPO.load(str(model_path), device="auto")

    env = make_env(args.stage)
    try:
        check_model_compatibility(model, env)
        start_time = time.time()
        results = evaluate_episodes(
            env=env,
            policy_fn=make_sb3_policy(model, deterministic=args.deterministic),
            episodes=args.episodes,
            seed=args.seed,
            stage=args.stage,
            verbose=True,
        )
        elapsed_time = time.time() - start_time
    finally:
        env.close()

    summary = summarize_results(results)
    metadata = {
        "algorithm": "PPO",
        "model": str(model_path),
        "stage": args.stage,
        "episodes": args.episodes,
        "seed": args.seed,
        "deterministic": bool(args.deterministic),
        "elapsed_time_seconds": elapsed_time,
    }

    save_evaluation_outputs(
        metadata=metadata,
        summary=summary,
        episode_rows=[result.to_row() for result in results],
        results_json=results_json,
        episodes_csv=episodes_csv,
        summary_json=summary_json,
    )

    print_summary(summary, title="Evaluation Summary")
    print("\nSaved evaluation outputs:")
    print(f"  Results JSON: {results_json}")
    print(f"  Episodes CSV: {episodes_csv}")
    print(f"  Summary JSON: {summary_json}")


def main() -> None:
    try:
        run()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from None


if __name__ == "__main__":
    main()
