from __future__ import annotations

import csv
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

ALL_STAGES=(
    "fixed_no_obstacles",
    "random_start_no_obstacles",
    "moving_no_obstacles",
    "fixed_obstacles",
    "moving_obstacles",
)


SUMMARY_COLUMNS=[
    "success_rate",
    "collision_rate",
    "out_of_bounds_rate",
    "timeouts_rate",
    "mean_return",
    "mean_final_distance",
    "mean_min_distance",
    "mean_episode_length",
]


PolicyFn=Callable[[Any,dict[str,Any], Any],int]
_ENV_CLASSES:tuple[Any,Any]|None=None


@dataclass
class EpisodeResult:
    episode:int
    episode_return:float
    episode_length:int
    success:bool
    collision:bool
    out_of_bounds:bool
    timeout:bool
    final_distance_to_target:float
    min_distance_to_target:float
    stage:str | None=None

    def to_row(self,*,include_stage:bool=False) -> dict[str, Any]:
        row:dict[str, Any]={}
        if include_stage:
            row["stage"]=self.stage

        row.update(
            {
            "episode":self.episode,
            "return":self.episode_return,
            "episode_length":self.episode_length,
            "success":self.success,
            "collision":self.collision,
            "out_of_bounds":self.out_of_bounds,
            "timeout":self.timeout,
            "final_distance_to_target":self.final_distance_to_target,
            "min_distance_to_target":self.min_distance_to_target,
            }
        )

        return row
    
def get_env_classes()-> tuple[Any,Any]:
    global _ENV_CLASSES
    if _ENV_CLASSES is None:
        try:
            from deepblue_auv_rl.envs.auv_target_env import AUVTargetEnv ,MissionConfig
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "Failed to import AUVTargetEnv and MissionConfig. "
                "Ensure that the deepblue_auv_rl package is installed correctly."
            ) from e
        
        _ENV_CLASSES=(AUVTargetEnv,MissionConfig)

    return _ENV_CLASSES

def build_env_config(stage:str)->Any:
    _,MissionConfig=get_env_classes()

    if stage=="fixed_no_obstacles":
        return MissionConfig(
            random_start=False,
            random_target=False,
            moving_target=False,
           obstacles_enabled=False,
        )
    if stage=="random_start_no_obstacles":
        return MissionConfig(
            random_start=True,
            random_target=False,
            moving_target=False,
            obstacles_enabled=False,
        )
    if stage=="moving_no_obstacles":
        return MissionConfig(
            random_start=True,
            random_target=False,
            moving_target=True,
            obstacles_enabled=False,
        )
    if stage=="fixed_obstacles":
        return MissionConfig(
            random_start=False,
            random_target=False,
            moving_target=False,
            obstacles_enabled=True,
            num_obstacles=1,
        )
    if stage=="moving_obstacles":
        return MissionConfig(
            random_start=True,
            random_target=False,
            moving_target=False,
            obstacles_enabled=True,
            num_obstacles=1,
            moving_obstacles=True,
        )
    raise ValueError(f"Unknown stage: {stage}. Valid stages are: {ALL_STAGES}")

def make_env(stage: str, *, show_viewport: bool = False, verbose: bool = False) -> Any:
    AUVTargetEnv, _ = get_env_classes()

    return AUVTargetEnv(
        config=build_env_config(stage),
        show_viewport=show_viewport,
        verbose=verbose,
        auto_start=False,
    )

def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"Expected a positive integer, got {value}")
    return parsed


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value

    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False

    raise ValueError(f"Expected a boolean value, got {value}")

def load_sb3_dependencies() -> tuple[Any, Any]:
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.utils import set_random_seed
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Stable-Baselines3 is required for PPO evaluation. Install the "
            "project development dependencies, for example: pip install -e .[dev]"
        ) from exc

    return PPO, set_random_seed

def resolve_model_path(
    raw_path: str,
    *,
    repo_root: Path,
    model_dir: Path,
) -> Path:
    requested_path = Path(raw_path).expanduser()
    search_roots = [Path()] if requested_path.is_absolute() else [repo_root, model_dir]

    candidates: list[Path] = []
    for root in search_roots:
        base = requested_path if requested_path.is_absolute() else root / requested_path
        candidates.append(base)
        if base.suffix != ".zip":
            candidates.append(base.with_suffix(".zip"))

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    checked = "\n  ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Model file not found. Checked:\n  {checked}")


def check_model_compatibility(model: Any, env: Any) -> None:
    model_obs_shape = getattr(model.observation_space, "shape", None)
    env_obs_shape = getattr(env.observation_space, "shape", None)
    if model_obs_shape != env_obs_shape:
        raise ValueError(
            "Model observation space shape does not match environment "
            f"observation space shape: {model_obs_shape} != {env_obs_shape}"
        )

    model_action_count = getattr(model.action_space, "n", None)
    env_action_count = getattr(env.action_space, "n", None)
    if model_action_count != env_action_count:
        raise ValueError(
            "Model action count does not match environment action count: "
            f"{model_action_count} != {env_action_count}"
        )


def predict_sb3_action(model: Any, observation: Any, *, deterministic: bool) -> int:
    action, _ = model.predict(observation, deterministic=deterministic)

    try:
        return int(action.item())
    except AttributeError:
        return int(action)
    except ValueError as exc:
        raise ValueError(f"Expected a single discrete action, got {action}") from exc


def make_sb3_policy(model: Any, *, deterministic: bool) -> PolicyFn:
    def policy_fn(observation: Any, info: dict[str, Any], env: Any) -> int:
        del info, env
        return predict_sb3_action(model, observation, deterministic=deterministic)

    return policy_fn


def get_distance(info: dict[str, Any], observation: Any) -> float:
    if "distance_to_target" in info:
        return float(info["distance_to_target"])
    return float(observation[3])


def run_episode(
    *,
    env: Any,
    policy_fn: PolicyFn,
    episode: int,
    seed: int,
    stage: str | None = None,
) -> EpisodeResult:
    env.action_space.seed(seed)
    observation, info = env.reset(seed=seed)

    episode_return = 0.0
    episode_length = 0
    initial_distance = get_distance(info, observation)
    final_distance = initial_distance
    min_distance = initial_distance

    success = False
    collision = False
    out_of_bounds = False
    timeout = False

    while True:
        action = int(policy_fn(observation, info, env))
        observation, reward, terminated, truncated, info = env.step(action)

        episode_length += 1
        episode_return += float(reward)

        current_distance = get_distance(info, observation)
        final_distance = current_distance
        min_distance = min(min_distance, current_distance)

        success = success or bool(info.get("success", False))
        collision = collision or bool(info.get("collision", False))
        out_of_bounds = out_of_bounds or bool(info.get("out_of_bounds", False))
        timeout = timeout or bool(info.get("timeout", False))

        if terminated or truncated:
            timeout = timeout or bool(
                truncated and not (success or collision or out_of_bounds)
            )
            break

    return EpisodeResult(
        stage=stage,
        episode=episode,
        episode_return=float(episode_return),
        episode_length=int(episode_length),
        success=bool(success),
        collision=bool(collision),
        out_of_bounds=bool(out_of_bounds),
        timeout=bool(timeout),
        final_distance_to_target=float(final_distance),
        min_distance_to_target=float(min_distance),
    )


def evaluate_episodes(
    *,
    env: Any,
    policy_fn: PolicyFn,
    episodes: int,
    seed: int,
    stage: str | None = None,
    verbose: bool = True,
) -> list[EpisodeResult]:
    results: list[EpisodeResult] = []

    for episode_idx in range(episodes):
        result = run_episode(
            env=env,
            policy_fn=policy_fn,
            episode=episode_idx + 1,
            seed=seed + episode_idx,
            stage=stage,
        )
        results.append(result)

        if verbose:
            print_episode_result(result)

    return results


def summarize_results(results: list[EpisodeResult]) -> dict[str, float | int]:
    if not results:
        raise ValueError("No episode results to summarize.")

    returns = [result.episode_return for result in results]
    lengths = [result.episode_length for result in results]
    final_distances = [result.final_distance_to_target for result in results]
    min_distances = [result.min_distance_to_target for result in results]
    episode_count = len(results)

    return {
        "episodes": int(episode_count),
        "success_rate": sum(result.success for result in results) / episode_count,
        "collision_rate": sum(result.collision for result in results) / episode_count,
        "out_of_bounds_rate": (
            sum(result.out_of_bounds for result in results) / episode_count
        ),
        "timeout_rate": sum(result.timeout for result in results) / episode_count,
        "mean_return": float(statistics.fmean(returns)),
        "std_return": float(statistics.pstdev(returns)),
        "mean_final_distance": float(statistics.fmean(final_distances)),
        "mean_min_distance": float(statistics.fmean(min_distances)),
        "mean_episode_length": float(statistics.fmean(lengths)),
    }


def summary_row(summary: dict[str, float | int], **prefix: Any) -> dict[str, Any]:
    return {
        **prefix,
        **{column: summary[column] for column in SUMMARY_COLUMNS},
    }


def print_episode_result(result: EpisodeResult) -> None:
    print(
        f"Episode {result.episode}: "
        f"return={result.episode_return:.2f}, "
        f"length={result.episode_length}, "
        f"success={result.success}, "
        f"collision={result.collision}, "
        f"out_of_bounds={result.out_of_bounds}, "
        f"timeout={result.timeout}, "
        f"final_distance={result.final_distance_to_target:.2f}, "
        f"min_distance={result.min_distance_to_target:.2f}"
    )


def print_summary(summary: dict[str, float | int], *, title: str) -> None:
    print(f"\n=== {title} ===")
    print(f"Episodes: {summary['episodes']}")
    print(f"Success Rate: {100.0 * summary['success_rate']:.2f}%")
    print(f"Collision Rate: {100.0 * summary['collision_rate']:.2f}%")
    print(f"Out-of-Bounds Rate: {100.0 * summary['out_of_bounds_rate']:.2f}%")
    print(f"Timeout Rate: {100.0 * summary['timeout_rate']:.2f}%")
    print(f"Mean Return: {summary['mean_return']:.3f}")
    if "std_return" in summary:
        print(f"Std Return: {summary['std_return']:.3f}")
    print(f"Mean Final Distance: {summary['mean_final_distance']:.3f}")
    print(f"Mean Minimum Distance: {summary['mean_min_distance']:.3f}")
    print(f"Mean Episode Length: {summary['mean_episode_length']:.3f}")


def save_evaluation_outputs(
    *,
    metadata: dict[str, Any],
    summary: dict[str, float | int],
    episode_rows: list[dict[str, Any]],
    results_json: Path,
    episodes_csv: Path,
    summary_json: Path,
) -> None:
    with results_json.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "metadata": metadata,
                "summary": summary,
                "episodes": episode_rows,
            },
            file,
            indent=2,
        )

    with episodes_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(episode_rows[0]))
        writer.writeheader()
        writer.writerows(episode_rows)

    with summary_json.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "metadata": metadata,
                "summary": summary,
            },
            file,
            indent=2,
        )


def resolve_output_dir(raw_output: str, *, repo_root: Path) -> Path:
    output_dir = Path(raw_output).expanduser()
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def baseline_output_paths(
    *,
    output_dir: Path,
    policy_name: str,
    stage: str,
    episodes: int,
    seed: int,
) -> tuple[Path, Path, Path]:
    stem = f"{policy_name}_{stage}_{episodes}eps_seed_{seed}"
    return (
        output_dir / f"{stem}.json",
        output_dir / f"{stem}_episodes.csv",
        output_dir / f"{stem}_summary.json",
    )


def save_baseline_summary_csv(
    *,
    output_dir: Path,
    filename: str,
    rows: list[dict[str, Any]],
    prefix_columns: list[str],
) -> Path:
    output_path = output_dir / filename
    fieldnames = [*prefix_columns, *SUMMARY_COLUMNS]

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return output_path



