#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from deepblue_auv_rl.evaluation.evaluate import (
    ALL_STAGES,
    check_model_compatibility,
    load_sb3_dependencies,
    make_env,
    positive_int,
    predict_sb3_action,
    resolve_model_path,
    resolve_output_dir,
    str_to_bool,
)


MODEL_DIR = REPO_ROOT / "models"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "rollouts"


def get_position(info: dict[str, Any]) -> list[float]:
    position = info.get("position")
    if position is None:
        raise ValueError("Environment info did not contain position.")
    return [float(position[0]), float(position[1]), float(position[2])]


def get_target_position(info: dict[str, Any]) -> list[float]:
    target_position = info.get("target_position")
    if target_position is None:
        raise ValueError("Environment info did not contain target_position.")
    return [
        float(target_position[0]),
        float(target_position[1]),
        float(target_position[2]),
    ]


def get_obstacle_positions(env: Any) -> list[list[float]]:
    config = getattr(env, "config", None)
    if config is None or not getattr(config, "obstacles_enabled", False):
        return []

    num_obstacles = int(getattr(config, "num_obstacles", 0))
    if num_obstacles <= 0:
        return []

    runtime_positions = getattr(env, "_obstacle_positions_runtime", None)
    if runtime_positions is not None:
        positions = runtime_positions
    else:
        positions = getattr(config, "obstacle_positions", ())[:num_obstacles]

    return [
        [float(position[0]), float(position[1]), float(position[2])]
        for position in positions[:num_obstacles]
    ]


def done_reason(
    *,
    info: dict[str, Any],
    terminated: bool,
    truncated: bool,
) -> str:
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


def make_row(
    *,
    timestep: int,
    info: dict[str, Any],
    env: Any,
    reward: float,
    done: bool,
    reason: str,
) -> dict[str, Any]:
    position = get_position(info)
    target_position = get_target_position(info)
    obstacle_positions = get_obstacle_positions(env)

    return {
        "timestep": int(timestep),
        "auv_x": position[0],
        "auv_y": position[1],
        "auv_z": position[2],
        "target_x": target_position[0],
        "target_y": target_position[1],
        "target_z": target_position[2],
        "obstacle_positions": obstacle_positions,
        "distance_to_target": float(info.get("distance_to_target", 0.0)),
        "reward": float(reward),
        "done": bool(done),
        "done_reason": reason,
        "action": info.get("action"),
        "action_name": info.get("action_name"),
    }


def episode_done_reason(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "unknown"

    reason = str(rows[-1].get("done_reason", "") or "")
    if reason:
        return reason

    if bool(rows[-1].get("done", False)):
        return "done"
    return "incomplete"


def flatten_trajectory_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    max_obstacles = max(
        (len(row["obstacle_positions"]) for row in rows),
        default=0,
    )

    flattened_rows: list[dict[str, Any]] = []
    for row in rows:
        flat_row = dict(row)
        obstacle_positions = flat_row.pop("obstacle_positions")
        flat_row["obstacle_positions_json"] = json.dumps(obstacle_positions)

        for obstacle_idx in range(max_obstacles):
            if obstacle_idx < len(obstacle_positions):
                obstacle = obstacle_positions[obstacle_idx]
                flat_row[f"obstacle_{obstacle_idx}_x"] = obstacle[0]
                flat_row[f"obstacle_{obstacle_idx}_y"] = obstacle[1]
                flat_row[f"obstacle_{obstacle_idx}_z"] = obstacle[2]
            else:
                flat_row[f"obstacle_{obstacle_idx}_x"] = ""
                flat_row[f"obstacle_{obstacle_idx}_y"] = ""
                flat_row[f"obstacle_{obstacle_idx}_z"] = ""

        flattened_rows.append(flat_row)

    return flattened_rows


def save_trajectory_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    flattened_rows = flatten_trajectory_rows(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(flattened_rows[0]))
        writer.writeheader()
        writer.writerows(flattened_rows)


def plot_trajectory(
    *,
    rows: list[dict[str, Any]],
    output_base: Path,
    title: str,
) -> None:
    auv_x = [float(row["auv_x"]) for row in rows]
    auv_y = [float(row["auv_y"]) for row in rows]
    target_x = [float(row["target_x"]) for row in rows]
    target_y = [float(row["target_y"]) for row in rows]

    fig, ax = plt.subplots(figsize=(8.0, 7.0))
    ax.plot(auv_x, auv_y, color="#1f77b4", linewidth=2.0, label="AUV path")
    ax.scatter(auv_x[0], auv_y[0], color="#2ca02c", s=80, zorder=4, label="Start")
    ax.scatter(auv_x[-1], auv_y[-1], color="#d62728", s=80, zorder=4, label="Final")

    if len(set(zip(target_x, target_y))) > 1:
        ax.plot(
            target_x,
            target_y,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label="Target path",
        )
        ax.scatter(
            target_x[-1],
            target_y[-1],
            color="black",
            marker="*",
            s=140,
            zorder=5,
            label="Final target",
        )
    else:
        ax.scatter(
            target_x[-1],
            target_y[-1],
            color="black",
            marker="*",
            s=140,
            zorder=5,
            label="Target",
        )

    max_obstacles = max(
        (len(row["obstacle_positions"]) for row in rows),
        default=0,
    )
    for obstacle_idx in range(max_obstacles):
        obstacle_points = [
            row["obstacle_positions"][obstacle_idx]
            for row in rows
            if obstacle_idx < len(row["obstacle_positions"])
        ]
        if not obstacle_points:
            continue

        obstacle_x = [point[0] for point in obstacle_points]
        obstacle_y = [point[1] for point in obstacle_points]
        label = f"Obstacle {obstacle_idx}"

        if len(set(zip(obstacle_x, obstacle_y))) > 1:
            ax.plot(
                obstacle_x,
                obstacle_y,
                color="#ff7f0e",
                linestyle=":",
                linewidth=1.5,
                label=f"{label} path",
            )
            ax.scatter(
                obstacle_x[-1],
                obstacle_y[-1],
                color="#ff7f0e",
                marker="X",
                s=90,
                zorder=4,
                label=f"{label} final",
            )
        else:
            ax.scatter(
                obstacle_x[-1],
                obstacle_y[-1],
                color="#ff7f0e",
                marker="X",
                s=90,
                zorder=4,
                label=label,
            )

    ax.set_title(title)
    ax.set_xlabel("x position")
    ax.set_ylabel("y position")
    ax.grid(alpha=0.25)
    ax.axis("equal")
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()

    for suffix in ("png", "pdf"):
        output_path = output_base.with_suffix(f".{suffix}")
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot: {output_path}")

    plt.close(fig)


def rollout_episode(
    *,
    env: Any,
    model: Any,
    episode_seed: int,
    deterministic: bool,
) -> list[dict[str, Any]]:
    observation, info = env.reset(seed=episode_seed)
    rows = [
        make_row(
            timestep=0,
            info=info,
            env=env,
            reward=0.0,
            done=False,
            reason="",
        )
    ]

    timestep = 0
    while True:
        action = predict_sb3_action(model, observation, deterministic=deterministic)
        observation, reward, terminated, truncated, info = env.step(action)
        timestep += 1

        done = bool(terminated or truncated)
        reason = done_reason(
            info=info,
            terminated=terminated,
            truncated=truncated,
        )
        rows.append(
            make_row(
                timestep=timestep,
                info=info,
                env=env,
                reward=float(reward),
                done=done,
                reason=reason,
            )
        )

        if done:
            break

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save PPO rollout trajectories and top-down trajectory plots."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model checkpoint path. Bare names are resolved inside models/.",
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=ALL_STAGES,
        help="Mission stage/environment configuration to run.",
    )
    parser.add_argument(
        "--episodes",
        type=positive_int,
        default=1,
        help="Number of rollout episodes to save.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed. Episode i uses seed + i.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR.relative_to(REPO_ROOT)),
        help="Directory where trajectory CSVs and plots are saved.",
    )
    parser.add_argument(
        "--deterministic",
        nargs="?",
        const=True,
        default=False,
        type=str_to_bool,
        help="Use deterministic PPO actions. Can be passed as a flag or true/false.",
    )
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    model_path = resolve_model_path(
        args.model,
        repo_root=REPO_ROOT,
        model_dir=MODEL_DIR,
    )
    output_dir = resolve_output_dir(args.output_dir, repo_root=REPO_ROOT)

    PPO, set_random_seed = load_sb3_dependencies()
    set_random_seed(args.seed)

    print("Loading PPO model:")
    print(f"  {model_path}")
    model = PPO.load(str(model_path), device="auto")

    env = make_env(args.stage)
    try:
        check_model_compatibility(model, env)

        for episode_idx in range(args.episodes):
            episode_number = episode_idx + 1
            episode_seed = args.seed + episode_idx
            rows = rollout_episode(
                env=env,
                model=model,
                episode_seed=episode_seed,
                deterministic=args.deterministic,
            )

            reason = episode_done_reason(rows)
            output_stem = (
                f"trajectory_{model_path.stem}_{args.stage}_"
                f"ep{episode_number:02d}_seed_{episode_seed}_{reason}"
            )
            csv_path = output_dir / f"{output_stem}.csv"
            plot_base = output_dir / output_stem

            save_trajectory_csv(rows, csv_path)
            print(f"Saved trajectory CSV: {csv_path}")

            plot_trajectory(
                rows=rows,
                output_base=plot_base,
                title=(
                    f"{model_path.stem}\n"
                    f"{args.stage}, episode {episode_number}, reason: {reason}"
                ),
            )
    finally:
        env.close()


def main() -> None:
    try:
        run()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from None


if __name__ == "__main__":
    main()
