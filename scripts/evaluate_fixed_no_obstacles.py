#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import time 
from dataclasses import asdict
from typing import Any
from pathlib import Path

from stable_baselines3 import PPO

from deepblue_auv_rl.envs.auv_target_env import AUVTargetEnv, MissionConfig

from deepblue_auv_rl.evaluation.metrics import (
    EpiosdeStats,
    evaluate_policy,
    make_sb3_policy,
    print_evaluation_summary,
    random_policy,
    summarize_episode_stats,
)
from train_ppo import build_env_config


REPO_ROOT= Path(__file__).resolve().parents[1]
RESULT_DIR= REPO_ROOT / "results"/"evaluation"
POLICY_NAMES=("stop","random","ppo")


def positive_int(value:str)-> int:

    parsed=int(value)
    if parsed<=0:
        raise argparse.ArgumentTypeError(f"Invalid positive integer value: {value}")
    return parsed

def resolve_model_path(raw_path:str)-> Path:
    """Resolve the model path, checking if it exists."""
    model_path=Path(raw_path).expanduser()

    if not model_path.is_absolute():
        model_path=REPO_ROOT/model_path

    if model_path.suffix!=".zip":
        model_path=model_path.with_suffix(".zip")

    model_path=model_path.resolve()

    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return model_path

def make_env(stage:str, show_viewport:bool=False)-> AUVTargetEnv:

    return AUVTargetEnv(
        config=build_env_config(stage),
        show_viewport=show_viewport,
        verbose=False,
        auto_start=False,
    )

def stop_policy(observation: Any,info: dict[str, Any],env:Any)-> int:

    del observation,info,env
    return 5 

def check_model_compatibility(model:PPO, env:AUVTargetEnv   )-> None:

    model_observation_shape=model.observation_space.shape
    env_observation_shape=env.observation_space.shape
    if model_observation_shape!=env_observation_shape:
        raise ValueError(
            f"Model observation space shape {model_observation_shape} does not match environment observation space shape {env_observation_shape}."
        )
    
    model_action_count=getattr(model.action_space,"n",None)
    env_action_count=getattr(env.action_space,"n",None)

    if model_action_count!=env_action_count:
        raise ValueError(
            f"Model action space count {model_action_count} does not match environment action space count {env_action_count}."
        )
    

def evaluate_one_policy(
    *,
    policy_name: str,
    model: PPO,
    stage: str,
    episodes: int,
    seed: int,
    max_steps: int | None,
    deterministic: bool,
    show_viewport: bool,
) -> tuple[list[EpisodeStats], dict[str, Any]]:
    env = make_env(stage, show_viewport=show_viewport)
    env.action_space.seed(seed)

    if policy_name == "stop":
        policy_fn = stop_policy
    elif policy_name == "random":
        policy_fn = random_policy
    elif policy_name == "ppo":
        policy_fn = make_sb3_policy(model, deterministic=deterministic)
    else:
        raise ValueError(f"Unknown policy: {policy_name}")

    try:
        stats = evaluate_policy(
            env=env,
            policy_fn=policy_fn,
            n_episodes=episodes,
            max_steps=max_steps,
            seed=seed,
            verbose=True,
        )
    finally:
        env.close()

    return stats, summarize_episode_stats(stats)


def save_results(
    *,
    model_path: Path,
    stage: str,
    episodes: int,
    seed: int,
    deterministic: bool,
    elapsed_time: float,
    summaries: dict[str, dict[str, Any]],
    episode_rows: list[dict[str, Any]],
) -> tuple[Path, Path]:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    prediction_mode = "deterministic" if deterministic else "stochastic"
    output_name = (
        f"eval_{model_path.stem}_{stage}_{episodes}eps_"
        f"seed_{seed}_{prediction_mode}"
    )
    json_path = RESULT_DIR / f"{output_name}.json"
    csv_path = RESULT_DIR / f"{output_name}.csv"

    output = {
        "model": str(model_path),
        "stage": stage,
        "episodes_per_policy": episodes,
        "seed": seed,
        "ppo_deterministic": deterministic,
        "policies": list(POLICY_NAMES),
        "summary": summaries,
        "episodes": episode_rows,
        "elapsed_time_seconds": elapsed_time,
    }

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(episode_rows[0]))
        writer.writeheader()
        writer.writerows(episode_rows)

    return json_path, csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a PPO checkpoint against stop and random baselines."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Checkpoint path, relative to the repository or absolute.",
    )
    parser.add_argument(
        "--stage",
        default="fixed_no_obstacles",
        choices=[
            "fixed_no_obstacles",
            "random_start_no_obstacles",
            "moving_no_obstacles",
            "fixed_obstacles",
            "moving_obstacles",
        ],
        help="Use the stage on which the model was trained for the main evaluation.",
    )
    parser.add_argument("--episodes", type=positive_int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-steps",
        type=positive_int,
        default=None,
        help="Optional evaluation limit; defaults to MissionConfig.max_steps.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample PPO actions instead of using deterministic predictions.",
    )
    parser.add_argument(
        "--show-viewport",
        action="store_true",
        help="Show the simulator viewport (slower).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = resolve_model_path(args.model)
    model = PPO.load(str(model_path), device=args.device)
    deterministic = not args.stochastic

    compatibility_env = make_env(args.stage)
    try:
        check_model_compatibility(model, compatibility_env)
    finally:
        compatibility_env.close()

    summaries: dict[str, dict[str, Any]] = {}
    episode_rows: list[dict[str, Any]] = []
    start_time = time.time()

    for policy_name in POLICY_NAMES:
        print(f"\n{'=' * 60}")
        print(f"Evaluating {policy_name} policy")
        print("=" * 60)

        stats, summary = evaluate_one_policy(
            policy_name=policy_name,
            model=model,
            stage=args.stage,
            episodes=args.episodes,
            seed=args.seed,
            max_steps=args.max_steps,
            deterministic=deterministic,
            show_viewport=args.show_viewport,
        )
        summaries[policy_name] = summary
        episode_rows.extend(
            {"policy": policy_name, **asdict(episode)} for episode in stats
        )

        print_evaluation_summary(summary, title=f"{policy_name.upper()} Policy")
        print(f"Collision Rate: {summary['collision_rate_percent']:.2f}%")
        print(
            "Average Safe-Distance Violations: "
            f"{summary['average_safe_distance_violations']:.2f}"
        )

    json_path, csv_path = save_results(
        model_path=model_path,
        stage=args.stage,
        episodes=args.episodes,
        seed=args.seed,
        deterministic=deterministic,
        elapsed_time=time.time() - start_time,
        summaries=summaries,
        episode_rows=episode_rows,
    )

    print("\nEvaluation completed.")
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")


if __name__ == "__main__":
    main()






 

    

    


