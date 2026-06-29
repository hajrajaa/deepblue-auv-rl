#!/usr/bin/env python3

# Example:
# python scripts/sample_model.py \
#   --model models/ppo_curriculum_moving_obstacles_100000_steps_seed_42.zip \
#   --stage moving_obstacles \
#   --episodes 1 \
#   --seed 2000 \
#   --deterministic \
#   --render \
#   --visual-markers \
#   --save-trajectory \
#   --trajectory-output outputs/demo/trajectory_moving_obstacles.png \
#   --sleep 0.05

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
    make_env,
    positive_int,
    resolve_model_path,
    str_to_bool,
)


MODEL_DIR = REPO_ROOT / "models"
DEFAULT_TRAJECTORY_DIR = REPO_ROOT / "outputs" / "demo"
IMAGE_SUFFIXES = {".png", ".pdf", ".svg"}


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
            "Stable-Baselines3 is required to sample PPO models. Install the "
            "project dependencies, for example: pip install -e .[dev]"
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


def as_xyz(value: Any) -> list[float] | None:
    if value is None:
        return None

    try:
        return [float(value[0]), float(value[1]), float(value[2])]
    except (TypeError, IndexError, KeyError, ValueError):
        return None


def format_vector(value: Any) -> str:
    position = as_xyz(value)
    if position is None:
        return "n/a"
    return f"[{position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}]"


def format_optional_float(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def extract_positions_from_collection(raw_positions: Any) -> list[list[float]]:
    if raw_positions is None:
        return []

    if isinstance(raw_positions, dict):
        iterable = raw_positions.values()
    else:
        iterable = raw_positions

    positions: list[list[float]] = []
    try:
        for item in iterable:
            position = as_xyz(item)
            if position is None and isinstance(item, dict):
                for key in ("position", "location", "pos"):
                    position = as_xyz(item.get(key))
                    if position is not None:
                        break
            if position is None:
                for attr in ("position", "location", "pos"):
                    position = as_xyz(getattr(item, attr, None))
                    if position is not None:
                        break
            if position is not None:
                positions.append(position)
    except TypeError:
        position = as_xyz(raw_positions)
        if position is not None:
            positions.append(position)

    return positions


def read_demo_state(env: Any, info: dict[str, Any]) -> dict[str, Any]:
    state: dict[str, Any] = {}
    get_demo_state = getattr(env, "get_demo_state", None)
    if callable(get_demo_state):
        state = dict(get_demo_state())

    get_auv_position = getattr(env, "get_auv_position", None)
    auv_position = as_xyz(state.get("auv_position"))
    if auv_position is None and callable(get_auv_position):
        auv_position = as_xyz(get_auv_position())
    if auv_position is None:
        auv_position = as_xyz(info.get("position"))

    get_target_position = getattr(env, "get_target_position", None)
    target_position = as_xyz(state.get("target_position"))
    if target_position is None and callable(get_target_position):
        target_position = as_xyz(get_target_position())
    if target_position is None:
        target_position = as_xyz(info.get("target_position"))
    if target_position is None:
        for attr in ("target_position", "target_pos", "goal_position", "goal_pos"):
            target_position = as_xyz(getattr(env, attr, None))
            if target_position is not None:
                break

    get_obstacle_positions = getattr(env, "get_obstacle_positions", None)
    obstacle_positions = extract_positions_from_collection(
        state.get("obstacle_positions")
    )
    if not obstacle_positions and callable(get_obstacle_positions):
        obstacle_positions = extract_positions_from_collection(get_obstacle_positions())
    if not obstacle_positions:
        for attr in (
            "_obstacle_positions_runtime",
            "obstacle_positions",
            "obstacles",
            "fixed_obstacles",
            "moving_obstacles",
        ):
            obstacle_positions = extract_positions_from_collection(
                getattr(env, attr, None)
            )
            if obstacle_positions:
                break

    config = getattr(env, "config", None)
    if not obstacle_positions and config is not None:
        num_obstacles = int(getattr(config, "num_obstacles", 0))
        if bool(getattr(config, "obstacles_enabled", False)) and num_obstacles > 0:
            obstacle_positions = extract_positions_from_collection(
                getattr(config, "obstacle_positions", ())[:num_obstacles]
            )

    return {
        "auv_position": auv_position,
        "target_position": target_position,
        "obstacle_positions": obstacle_positions,
        "moving_obstacles": bool(
            state.get("moving_obstacles", getattr(config, "moving_obstacles", False))
        )
        if config is not None
        else bool(state.get("moving_obstacles", False)),
        "obstacle_radius": float(
            state.get("obstacle_radius", getattr(config, "obstacle_radius", 1.0))
        )
        if config is not None
        else float(state.get("obstacle_radius", 1.0)),
    }


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


def make_trajectory_row(
    *,
    timestep: int,
    env: Any,
    info: dict[str, Any],
    reward: float,
    action: int | None,
    done: bool,
    reason: str,
) -> dict[str, Any]:
    demo_state = read_demo_state(env, info)
    auv_position = demo_state["auv_position"]
    target_position = demo_state["target_position"]

    if auv_position is None:
        raise ValueError("Could not read the real AUV position from env state/info.")
    if target_position is None:
        raise ValueError("Could not read target position from env state/info.")

    return {
        "timestep": int(timestep),
        "auv_position": auv_position,
        "target_position": target_position,
        "obstacle_positions": demo_state["obstacle_positions"],
        "moving_obstacles": bool(demo_state["moving_obstacles"]),
        "obstacle_radius": float(demo_state["obstacle_radius"]),
        "distance_to_target": info.get("distance_to_target"),
        "reward": float(reward),
        "action": action,
        "done": bool(done),
        "done_reason": reason,
        "success": bool(info.get("success", False)),
        "collision": bool(info.get("collision", False)),
        "out_of_bounds": bool(info.get("out_of_bounds", False)),
        "timeout": bool(info.get("timeout", False)),
    }


class DemoMarkerManager:
    def __init__(self, env: Any, enabled: bool, save_trajectory: bool) -> None:
        self.env = env
        self.enabled = enabled
        self.save_trajectory = save_trajectory
        self._warned = False
        self._announced = False
        self._draw_owner: Any | None = None
        self._draw_method_name: str | None = None

    def update(self, row: dict[str, Any]) -> None:
        if not self.enabled:
            return

        target_position = row["target_position"]
        obstacle_positions = row["obstacle_positions"]
        moving_obstacles = bool(row["moving_obstacles"])

        target_ok = self._draw_marker(
            location=target_position,
            color=(0, 255, 0),
            radius=0.8,
            label="target",
        )
        obstacle_ok = True
        obstacle_color = (255, 165, 0) if moving_obstacles else (255, 0, 0)
        for obstacle_position in obstacle_positions:
            obstacle_ok = (
                self._draw_marker(
                    location=obstacle_position,
                    color=obstacle_color,
                    radius=max(float(row.get("obstacle_radius", 1.0)), 0.4),
                    label="obstacle",
                )
                and obstacle_ok
            )

        if target_ok and obstacle_ok and not self._announced:
            print(
                "Visual markers: using HoloOcean debug draw API "
                f"{self._draw_method_name}."
            )
            self._announced = True
        elif not target_ok:
            self._warn_fallback()

    def _warn_fallback(self) -> None:
        if self._warned:
            return

        message = (
            "Warning: HoloOcean visual marker API was not available through "
            "this environment. Target and obstacles will still be shown in the "
            "2D trajectory plot."
        )
        if not self.save_trajectory:
            message += " Pass --save-trajectory to write that fallback plot."
        print(message)
        self._warned = True

    def _draw_marker(
        self,
        *,
        location: list[float],
        color: tuple[int, int, int],
        radius: float,
        label: str,
    ) -> bool:
        if self._draw_owner is not None and self._draw_method_name is not None:
            method = getattr(self._draw_owner, self._draw_method_name, None)
            if callable(method):
                return self._call_draw_method(
                    method,
                    location=location,
                    color=color,
                    radius=radius,
                    label=label,
                )

        holo_env = getattr(self.env, "_holo_env", None)
        if holo_env is None:
            return False

        owners = [holo_env]
        client = getattr(holo_env, "client", None)
        if client is not None:
            owners.append(client)

        method_names = (
            "draw_debug_sphere",
            "draw_sphere",
            "draw_debug_point",
            "draw_point",
            "add_debug_sphere",
        )
        for owner in owners:
            for method_name in method_names:
                method = getattr(owner, method_name, None)
                if not callable(method):
                    continue
                if self._call_draw_method(
                    method,
                    location=location,
                    color=color,
                    radius=radius,
                    label=label,
                ):
                    self._draw_owner = owner
                    self._draw_method_name = method_name
                    return True

        return False

    def _call_draw_method(
        self,
        method: Any,
        *,
        location: list[float],
        color: tuple[int, int, int],
        radius: float,
        label: str,
    ) -> bool:
        color_list = [int(color[0]), int(color[1]), int(color[2])]
        attempts = (
            {
                "location": location,
                "color": color_list,
                "radius": radius,
                "life_time": 0.25,
                "label": label,
            },
            {
                "position": location,
                "color": color_list,
                "radius": radius,
                "life_time": 0.25,
            },
            {
                "loc": location,
                "color": color_list,
                "size": radius,
                "life_time": 0.25,
            },
        )

        for kwargs in attempts:
            try:
                method(**kwargs)
                return True
            except TypeError:
                pass
            except Exception:
                return False

        positional_attempts = (
            (location, color_list, radius),
            (location, radius, color_list),
            (location, color_list),
            (location,),
        )
        for args in positional_attempts:
            try:
                method(*args)
                return True
            except TypeError:
                pass
            except Exception:
                return False

        return False


def resolve_trajectory_output_path(
    *,
    raw_output: str | None,
    model_path: Path,
    stage: str,
    episode: int,
    total_episodes: int,
    seed: int,
) -> Path:
    default_name = (
        f"trajectory_{model_path.stem}_{stage}_ep{episode:02d}_seed_{seed}.png"
    )

    if raw_output is None:
        return DEFAULT_TRAJECTORY_DIR / default_name

    output_path = Path(raw_output).expanduser()
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path

    if output_path.suffix.lower() not in IMAGE_SUFFIXES:
        return output_path / default_name

    if total_episodes <= 1:
        return output_path

    return output_path.with_name(
        f"{output_path.stem}_ep{episode:02d}{output_path.suffix}"
    )


def plot_trajectory(
    *,
    rows: list[dict[str, Any]],
    output_path: Path,
    stage: str,
    episode: int,
    total_episodes: int,
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib is required to save trajectory plots."
        ) from exc

    if not rows:
        raise ValueError("No trajectory rows were recorded.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    auv_x = [row["auv_position"][0] for row in rows]
    auv_y = [row["auv_position"][1] for row in rows]
    target_x = [row["target_position"][0] for row in rows]
    target_y = [row["target_position"][1] for row in rows]

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    ax.plot(auv_x, auv_y, color="#1f77b4", linewidth=2.4, label="AUV path")
    ax.scatter(auv_x[0], auv_y[0], color="#2ca02c", s=90, zorder=5, label="Start")
    ax.scatter(auv_x[-1], auv_y[-1], color="#d62728", s=90, zorder=5, label="Final")

    if len(set(zip(target_x, target_y))) > 1:
        ax.plot(
            target_x,
            target_y,
            color="#2ca02c",
            linestyle="--",
            linewidth=1.6,
            label="Target path",
        )
        target_label = "Final target"
    else:
        target_label = "Target"

    ax.scatter(
        target_x[-1],
        target_y[-1],
        color="#2ca02c",
        marker="*",
        edgecolors="black",
        linewidths=0.8,
        s=180,
        zorder=6,
        label=target_label,
    )

    max_obstacles = max((len(row["obstacle_positions"]) for row in rows), default=0)
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
        obstacle_moved = len(set(zip(obstacle_x, obstacle_y))) > 1
        color = "#ff7f0e" if obstacle_moved else "#d62728"
        label = f"Obstacle {obstacle_idx + 1}"

        if obstacle_moved:
            ax.plot(
                obstacle_x,
                obstacle_y,
                color=color,
                linestyle=":",
                linewidth=1.8,
                label=f"{label} path",
            )
            label = f"{label} final"

        ax.scatter(
            obstacle_x[-1],
            obstacle_y[-1],
            color=color,
            marker="X",
            s=110,
            zorder=5,
            label=label,
        )

        radius = float(rows[-1].get("obstacle_radius", 0.0))
        if radius > 0.0:
            ax.add_patch(
                Circle(
                    (obstacle_x[-1], obstacle_y[-1]),
                    radius=radius,
                    fill=False,
                    edgecolor=color,
                    alpha=0.35,
                    linewidth=1.2,
                )
            )

    final_row = rows[-1]
    final_distance = format_optional_float(final_row.get("distance_to_target"))
    title = (
        f"{stage} episode {episode}/{total_episodes} "
        f"success={final_row['success']} "
        f"collision={final_row['collision']} "
        f"out_of_bounds={final_row['out_of_bounds']} "
        f"timeout={final_row['timeout']}\n"
        f"final distance to target={final_distance}"
    )

    ax.set_title(title)
    ax.set_xlabel("x position")
    ax.set_ylabel("y position")
    ax.grid(alpha=0.25)
    ax.axis("equal")
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved trajectory plot to: {output_path}")


def print_step_info(
    *,
    row: dict[str, Any],
    action: int | None,
) -> None:
    print(
        f"step={row['timestep']:04d} "
        f"reward={row['reward']:.3f} "
        f"distance={format_optional_float(row['distance_to_target'])} "
        f"action={action if action is not None else 'n/a'} "
        f"success={row['success']} "
        f"collision={row['collision']} "
        f"out_of_bounds={row['out_of_bounds']} "
        f"timeout={row['timeout']}"
    )
    print(f"  AUV position: {format_vector(row['auv_position'])}")
    print(f"  Target position: {format_vector(row['target_position'])}")
    print(f"  Obstacle count: {len(row['obstacle_positions'])}")


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
        description=(
            "Run a trained Stable-Baselines3 PPO model in the HoloOcean "
            "AUVTargetEnv for visual or non-visual sampling."
        )
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
        help="Number of episodes to run.",
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
        "--render",
        action="store_true",
        help="Show the HoloOcean viewport instead of running headless.",
    )
    parser.add_argument(
        "--sleep",
        type=non_negative_float,
        default=0.0,
        help="Seconds to sleep after each environment step.",
    )
    parser.add_argument(
        "--visual-markers",
        action="store_true",
        help="Try to draw demo-only target/obstacle markers in HoloOcean.",
    )
    parser.add_argument(
        "--save-trajectory",
        action="store_true",
        help="Save a top-down trajectory plot at the end of each episode.",
    )
    parser.add_argument(
        "--trajectory-output",
        default=None,
        help=(
            "Trajectory plot file or output directory. Defaults to outputs/demo/. "
            "For multiple episodes, _epXX is appended to file names."
        ),
    )
    return parser.parse_args()


def run_episode(
    *,
    env: Any,
    model: Any,
    model_path: Path,
    stage: str,
    episode: int,
    total_episodes: int,
    seed: int,
    deterministic: bool,
    sleep_seconds: float,
    visual_markers: bool,
    save_trajectory: bool,
    trajectory_output: str | None,
) -> dict[str, Any]:
    env.action_space.seed(seed)
    obs, info = env.reset(seed=seed)

    print(f"\nEpisode {episode}/{total_episodes}")
    print(f"seed={seed}")

    marker_manager = DemoMarkerManager(
        env=env,
        enabled=visual_markers,
        save_trajectory=save_trajectory,
    )

    rows: list[dict[str, Any]] = []
    initial_row = make_trajectory_row(
        timestep=0,
        env=env,
        info=info,
        reward=0.0,
        action=None,
        done=False,
        reason="",
    )
    rows.append(initial_row)
    print_step_info(row=initial_row, action=None)
    marker_manager.update(initial_row)

    episode_return = 0.0
    step = 0

    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        action_int = action_to_int(action)

        obs, reward, terminated, truncated, info = env.step(action_int)
        done = bool(terminated or truncated)
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
        reason = done_reason(
            info=step_info,
            terminated=bool(terminated),
            truncated=bool(truncated),
        )

        row = make_trajectory_row(
            timestep=step,
            env=env,
            info=step_info,
            reward=float(reward),
            action=action_int,
            done=done,
            reason=reason,
        )
        rows.append(row)
        print_step_info(row=row, action=action_int)
        marker_manager.update(row)

        if sleep_seconds > 0.0:
            time.sleep(sleep_seconds)

        if done:
            print(
                f"Episode {episode} finished: "
                f"return={episode_return:.3f} "
                f"steps={step} "
                f"terminated={terminated} "
                f"truncated={truncated} "
                f"success={row['success']} "
                f"collision={row['collision']} "
                f"out_of_bounds={row['out_of_bounds']} "
                f"timeout={row['timeout']}"
            )

            if save_trajectory:
                output_path = resolve_trajectory_output_path(
                    raw_output=trajectory_output,
                    model_path=model_path,
                    stage=stage,
                    episode=episode,
                    total_episodes=total_episodes,
                    seed=seed,
                )
                plot_trajectory(
                    rows=rows,
                    output_path=output_path,
                    stage=stage,
                    episode=episode,
                    total_episodes=total_episodes,
                )

            return {
                "return": episode_return,
                "steps": step,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "success": row["success"],
                "collision": row["collision"],
                "out_of_bounds": row["out_of_bounds"],
                "timeout": row["timeout"],
                "final_distance": row["distance_to_target"],
            }


def run() -> None:
    args = parse_args()
    model_path = resolve_model_path(
        args.model,
        repo_root=REPO_ROOT,
        model_dir=MODEL_DIR,
    )

    print("Loading PPO model:")
    print(f"  {model_path}")
    model = load_ppo_model(model_path)

    env = make_env(
        args.stage,
        show_viewport=bool(args.render),
        verbose=False,
    )

    try:
        check_model_compatibility(model, env)

        print("Sampling configuration:")
        print(f"  stage: {args.stage}")
        print(f"  episodes: {args.episodes}")
        print(f"  seed: {args.seed}")
        print(f"  deterministic: {bool(args.deterministic)}")
        print(f"  render viewport: {bool(args.render)}")
        print(f"  visual markers: {bool(args.visual_markers)}")
        print(f"  save trajectory: {bool(args.save_trajectory)}")
        print(f"  trajectory output: {args.trajectory_output or DEFAULT_TRAJECTORY_DIR}")
        print(f"  sleep per step: {args.sleep}")
        if args.visual_markers and not args.render:
            print("Warning: --visual-markers is most useful with --render enabled.")

        summaries = []
        for episode_idx in range(args.episodes):
            summaries.append(
                run_episode(
                    env=env,
                    model=model,
                    model_path=model_path,
                    stage=args.stage,
                    episode=episode_idx + 1,
                    total_episodes=args.episodes,
                    seed=args.seed + episode_idx,
                    deterministic=bool(args.deterministic),
                    sleep_seconds=float(args.sleep),
                    visual_markers=bool(args.visual_markers),
                    save_trajectory=bool(args.save_trajectory),
                    trajectory_output=args.trajectory_output,
                )
            )
    finally:
        close_env(env)

    print("\nSampling complete.")
    for episode_idx, summary in enumerate(summaries, start=1):
        print(
            f"Episode {episode_idx}: "
            f"return={summary['return']:.3f}, "
            f"steps={summary['steps']}, "
            f"success={summary['success']}, "
            f"collision={summary['collision']}, "
            f"out_of_bounds={summary['out_of_bounds']}, "
            f"timeout={summary['timeout']}, "
            f"final_distance={format_optional_float(summary['final_distance'])}"
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
