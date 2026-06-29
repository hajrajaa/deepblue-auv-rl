#!/usr/bin/env python3

# Visual sampling example for RunPod Desktop/Kasm:
# cd /workspace/deepblue-auv-rl
# source .venv/bin/activate
# export PYTHONPATH=$PWD/src:$PYTHONPATH
# export DISPLAY=:1
# python scripts/sample_model.py \
#   --model models/ppo_curriculum_new_fixed_obstacles_150000_steps_seed_42.zip \
#   --stage fixed_obstacles \
#   --episodes 1 \
#   --seed 1000 \
#   --deterministic \
#   --sleep 0.05

from __future__ import annotations

import argparse
import csv
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
    make_env,
    positive_int,
    resolve_model_path,
    str_to_bool,
)


MODEL_DIR = REPO_ROOT / "models"
VISUAL_ROLLOUT_DIR = REPO_ROOT / "outputs" / "visual_rollouts"


def non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError(f"Expected a non-negative float, got {value}")
    return parsed


def load_ppo_model(model_path: Path) -> Any:
    try:
        from stable_baselines3 import PPO
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Stable-Baselines3 is required to sample PPO models. Activate the "
            "project environment first, for example: source .venv/bin/activate"
        ) from exc

    return PPO.load(str(model_path), device="auto")


def action_to_int(action: Any) -> int:
    try:
        return int(action.item())
    except AttributeError:
        pass

    if isinstance(action, (list, tuple)):
        if not action:
            raise ValueError("PPO returned an empty action sequence.")
        return int(action[0])

    return int(action)


def optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def format_optional_float(value: Any) -> str:
    parsed = optional_float(value)
    if parsed is None:
        return "n/a"
    return f"{parsed:.3f}"


def done_reason(info: dict[str, Any], *, terminated: bool, truncated: bool) -> str:
    if bool(info.get("success", False)):
        return "success"
    if bool(info.get("collision", False)):
        return "collision"
    if bool(info.get("out_of_bounds", False)):
        return "out_of_bounds"
    if bool(info.get("timeout", False)) or truncated:
        return "timeout"
    if terminated:
        return "terminated"
    return ""


def make_log_path(
    *,
    model_path: Path,
    stage: str,
    episodes: int,
    seed: int,
    deterministic: bool,
) -> Path:
    VISUAL_ROLLOUT_DIR.mkdir(parents=True, exist_ok=True)
    mode = "deterministic" if deterministic else "stochastic"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return (
        VISUAL_ROLLOUT_DIR
        / f"visual_rollout_{model_path.stem}_{stage}_{episodes}eps_seed_{seed}_{mode}_{timestamp}.csv"
    )


def position_columns(prefix: str, value: Any) -> dict[str, float | str]:
    try:
        return {
            f"{prefix}_x": float(value[0]),
            f"{prefix}_y": float(value[1]),
            f"{prefix}_z": float(value[2]),
        }
    except (TypeError, IndexError, KeyError, ValueError):
        return {
            f"{prefix}_x": "",
            f"{prefix}_y": "",
            f"{prefix}_z": "",
        }


def make_log_row(
    *,
    episode: int,
    step: int,
    action: int | None,
    reward: float,
    terminated: bool,
    truncated: bool,
    info: dict[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "episode": episode,
        "step": step,
        "action": "" if action is None else action,
        "reward": reward,
        "distance_to_target": "" if info.get("distance_to_target") is None else info.get("distance_to_target"),
        "success": optional_bool(info.get("success")),
        "collision": optional_bool(info.get("collision")),
        "out_of_bounds": optional_bool(info.get("out_of_bounds")),
        "timeout": optional_bool(info.get("timeout")),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "done_reason": done_reason(
            info,
            terminated=terminated,
            truncated=truncated,
        ),
    }
    row.update(position_columns("auv", info.get("position")))
    row.update(position_columns("target", info.get("target_position")))
    return row


def print_rollout_step(row: dict[str, Any]) -> None:
    print(
        f"episode={row['episode']} "
        f"step={int(row['step']):04d} "
        f"action={row['action']} "
        f"reward={float(row['reward']):.3f} "
        f"distance={format_optional_float(row['distance_to_target'])} "
        f"success={row['success']} "
        f"collision={row['collision']} "
        f"out_of_bounds={row['out_of_bounds']} "
        f"timeout={row['timeout']}"
    )


def save_csv_log(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved visual rollout CSV: {output_path}")


def close_env(env: Any) -> None:
    close = getattr(env, "close", None)
    if not callable(close):
        return

    try:
        close()
    except Exception as exc:
        print(f"Warning: failed to close environment cleanly: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a trained PPO model in a visible HoloOcean viewport."
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
        help="Mission stage/environment configuration to sample.",
    )
    parser.add_argument(
        "--episodes",
        type=positive_int,
        default=1,
        help="Number of visual rollout episodes.",
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
        "--sleep",
        type=non_negative_float,
        default=0.05,
        help="Seconds to sleep after each environment step.",
    )
    return parser.parse_args()


def run_episode(
    *,
    env: Any,
    model: Any,
    episode: int,
    total_episodes: int,
    seed: int,
    deterministic: bool,
    sleep_seconds: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    env.action_space.seed(seed)
    obs, info = env.reset(seed=seed)

    print(f"\nEpisode {episode}/{total_episodes}")
    print(f"seed={seed}")

    rows: list[dict[str, Any]] = []
    initial_row = make_log_row(
        episode=episode,
        step=0,
        action=None,
        reward=0.0,
        terminated=False,
        truncated=False,
        info=info,
    )
    rows.append(initial_row)
    print_rollout_step(initial_row)

    episode_return = 0.0
    step = 0

    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        action_int = action_to_int(action)

        obs, reward, terminated, truncated, info = env.step(action_int)
        step += 1
        episode_return += float(reward)

        step_info = dict(info)
        step_info.setdefault(
            "timeout",
            bool(
                truncated
                and not step_info.get("success", False)
                and not step_info.get("collision", False)
                and not step_info.get("out_of_bounds", False)
            ),
        )

        row = make_log_row(
            episode=episode,
            step=step,
            action=action_int,
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=step_info,
        )
        rows.append(row)
        print_rollout_step(row)

        if sleep_seconds > 0.0:
            time.sleep(sleep_seconds)

        if terminated or truncated:
            summary = {
                "return": episode_return,
                "steps": step,
                "success": bool(step_info.get("success", False)),
                "collision": bool(step_info.get("collision", False)),
                "out_of_bounds": bool(step_info.get("out_of_bounds", False)),
                "timeout": bool(step_info.get("timeout", False)),
                "done_reason": done_reason(
                    step_info,
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                ),
                "final_distance": step_info.get("distance_to_target"),
            }
            print(
                f"Episode {episode} finished: "
                f"return={summary['return']:.3f} "
                f"steps={summary['steps']} "
                f"reason={summary['done_reason']} "
                f"final_distance={format_optional_float(summary['final_distance'])}"
            )
            return rows, summary


def run() -> None:
    args = parse_args()
    deterministic = bool(args.deterministic)
    model_path = resolve_model_path(
        args.model,
        repo_root=REPO_ROOT,
        model_dir=MODEL_DIR,
    )
    csv_path = make_log_path(
        model_path=model_path,
        stage=args.stage,
        episodes=args.episodes,
        seed=args.seed,
        deterministic=deterministic,
    )

    print("Loading PPO model:")
    print(f"  {model_path}")
    model = load_ppo_model(model_path)

    print("Creating visible HoloOcean environment:")
    print("  show_viewport=True")
    print("  headless=False")
    env = make_env(
        args.stage,
        show_viewport=True,
        verbose=False,
    )

    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    try:
        check_model_compatibility(model, env)

        print("Visual rollout configuration:")
        print(f"  stage: {args.stage}")
        print(f"  episodes: {args.episodes}")
        print(f"  seed: {args.seed}")
        print(f"  deterministic: {deterministic}")
        print(f"  sleep per step: {args.sleep}")
        print(f"  csv log: {csv_path}")

        for episode_idx in range(args.episodes):
            rows, summary = run_episode(
                env=env,
                model=model,
                episode=episode_idx + 1,
                total_episodes=args.episodes,
                seed=args.seed + episode_idx,
                deterministic=deterministic,
                sleep_seconds=float(args.sleep),
            )
            all_rows.extend(rows)
            summaries.append(summary)
    finally:
        close_env(env)
        save_csv_log(all_rows, csv_path)

    print("\nVisual sampling complete.")
    for episode_idx, summary in enumerate(summaries, start=1):
        print(
            f"Episode {episode_idx}: "
            f"return={summary['return']:.3f}, "
            f"steps={summary['steps']}, "
            f"reason={summary['done_reason']}, "
            f"success={summary['success']}, "
            f"collision={summary['collision']}, "
            f"out_of_bounds={summary['out_of_bounds']}, "
            f"timeout={summary['timeout']}"
        )


def main() -> None:
    try:
        run()
    except KeyboardInterrupt:
        raise SystemExit("\nInterrupted by user.") from None
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from None


if __name__ == "__main__":
    main()
