from __future__ import annotations

import json
from dataclasses import dataclass,asdict
from typing import Any, Callable
from pathlib import Path
import numpy as np


PolicyFn = Callable[[np.ndarray, dict[str, Any],Any],int]


@dataclass
class EpisodeStats:
    """ Store evaluation information for a single episode."""
    
    episode:int
    total_reward:float
    episode_length:int

    success:bool
    out_of_bounds:bool
    collision:bool

    final_distance:float
    min_distance:float
    max_distance:float

    min_obstacle_distance:float|None
    safe_distance_violations:int
    
    reached_target_step:int|None
    terminated:bool
    truncated:bool

def random_policy(observation: np.ndarray, info: dict[str, Any],env:Any) -> int:
    """A simple random policy """
    return int(env.action_space.sample())

def make_sb3_policy(model:Any,deterministic:bool=True)-> PolicyFn:
    """ Wraps a Stable-Baselines3 model so it can be used by evalute_policy"""
    def policy_fn(observation: np.ndarray,info:dict[str,Any] , env:Any)-> Any:

        action , _=model.predict(observation , deterministic=deterministic)

        if isinstance(action, np.ndarray) and action.size==1:
            return int(action.item())
        
        return action
    
    return policy_fn


def evaluate_policy(env: Any, policy_fn: PolicyFn, n_episodes: int=10
                    ,max_steps:int|None=None, seed:int=0,verbose:bool=False) -> list[EpisodeStats]:

    all_stats: list[EpisodeStats] = []

    for episode_idx in range (n_episodes):

        obs,info= env.reset(seed=seed+episode_idx)
        total_reward = 0.0
        episode_length = 0

        initial_distance = _get_distance(info,obs)
        max_distance = initial_distance
        min_distance = initial_distance

        success = False
        out_of_bounds = False
        collision = False
        min_obstacle_distance =float("inf")
        safe_distance_violations = 0
        reached_target_step :int|None=None
        terminated = False
        truncated = False

        while True:
            action=policy_fn(obs,info,env)

            obs, reward, terminated, truncated, info = env.step(action)

            episode_length += 1
            total_reward += reward

            curr_distance = _get_distance(info,obs)
            min_distance = min(min_distance, curr_distance)
            max_distance = max(max_distance, curr_distance)


            success =bool(info.get("success", False))
            out_of_bounds = bool(info.get("out_of_bounds", False))
            collision = collision or bool(info.get("collision", False))

            if "closest_obstacle_distance" in info:
                obstacle_distance = float(info["closest_obstacle_distance"])
                min_obstacle_distance = min(min_obstacle_distance, obstacle_distance)

                safe_distance=float(info.get("safe_distance", 2.0))
                if obstacle_distance < safe_distance:
                    safe_distance_violations += 1


            if success and reached_target_step is None:
                reached_target_step = episode_length

            if terminated or truncated:
                break


            if max_steps is not None and episode_length >= max_steps:
                truncated = True
                break

        final_distance = _get_distance(info,obs)
        episode_stats = EpisodeStats(
            episode=episode_idx+1,
            total_reward=float(total_reward),
            episode_length=int(episode_length),
            success=bool(success),
            out_of_bounds=bool(out_of_bounds),
            collision=bool(collision),
            final_distance=float(final_distance),
            min_distance=float(min_distance),
            max_distance=float(max_distance),
            min_obstacle_distance=None if np.isinf(min_obstacle_distance) else float(min_obstacle_distance),
            safe_distance_violations=int(safe_distance_violations),
            reached_target_step=reached_target_step,
            terminated=bool(terminated),
            truncated=bool(truncated)
        )

        all_stats.append(episode_stats)


        if verbose:
            print(f"Episode {episode_stats.episode}: reward={episode_stats.total_reward:.2f}, "
                  f"length={episode_stats.episode_length}, success={episode_stats.success}, "
                  f"final_distance={episode_stats.final_distance:.2f}")
            
    return all_stats

def summarize_episode_stats(stats: list[EpisodeStats]) -> dict[str, Any]:
   
    num_episodes = len(stats)
    if num_episodes == 0:
        raise ValueError("No episode stats available. Run at least one episode .")
    
    returns=np.array([e.total_reward for e in stats], dtype=np.float64)
    lengths=np.array([e.episode_length for e in stats], dtype=np.float64)
    successes=np.array([e.success for e in stats], dtype=np.float64)
    out_of_bounds=np.array([e.out_of_bounds for e in stats], dtype=np.float64)
    final_distances=np.array([e.final_distance for e in stats], dtype=np.float64)
    min_distances=np.array([e.min_distance for e in stats], dtype=np.float64)
    truncated=np.array([e.truncated for e in stats], dtype=np.float64)
    collisions=np.array([e.collision for e in stats], dtype=np.float64)
    safe_violations=np.array([e.safe_distance_violations for e in stats], dtype=np.float64)
    obstacle_distances=[
        e.min_obstacle_distance 
        for e in stats if e.min_obstacle_distance is not None
    ]

    summary:dict[str, float|int] = {
        "num_episodes": int(num_episodes),
        "success_rate": float(np.mean(successes)),
        "success_rate_percent": float(np.mean(successes)*100.0),

        "average_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),


        "average_episode_length": float(np.mean(lengths)),
        "std_episode_length": float(np.std(lengths)),

        "average_final_distance": float(np.mean(final_distances)),
        "std_final_distance": float(np.std(final_distances)),

        "average_min_distance": float(np.mean(min_distances)),
        "std_min_distance": float(np.std(min_distances)),
        
        "out_of_bounds_rate": float(np.mean(out_of_bounds)),
        "out_of_bounds_rate_percent": float(np.mean(out_of_bounds)*100.0),

        "collision_rate": float(np.mean(collisions)),
        "collision_rate_percent": float(np.mean(collisions)*100.0),

        "average_safe_distance_violations": float(np.mean(safe_violations)),
        "total_safe_distance_violations": int(np.sum(safe_violations)),

        "truncated_rate": float(np.mean(truncated)),
        "truncated_rate_percent": float(np.mean(truncated)*100.0),

        "best_return": float(np.max(returns)),
        "worst_return": float(np.min(returns)),
        "best_final_distance": float(np.min(final_distances)),
        "worst_final_distance": float(np.max(final_distances)),
    }

    if len(obstacle_distances) > 0:
        obstacle_distances_np=np.array(obstacle_distances, dtype=np.float64)
        summary["average_min_obstacle_distance"] = float(np.mean(obstacle_distances_np))
        summary["std_min_obstacle_distance"] = float(np.std(obstacle_distances_np))

    else:
        summary["average_min_obstacle_distance"] = None
        summary["std_min_obstacle_distance"] = None
    return summary

def print_evaluation_summary(summary: dict[str, float|int], title: str="Evaluation Summary") -> None:

    print(f"\n=== {title} ===")

    print(f"Number of Episodes: {summary['num_episodes']}")
    print(f"Success Rate: {summary['success_rate_percent']:.2f}%")
    print(f"Average Return: {summary['average_return']:.3f}")
    print(f"Return Std : {summary['std_return']:.3f}")
    print(f"Average Episode Length: {summary['average_episode_length']:.3f}")
    print(f"Average Final Distance: {summary['average_final_distance']:.3f}")
    print(f"Average Min Distance: {summary['average_min_distance']:.3f}")
    print(f"Out of Bounds Rate: {summary['out_of_bounds_rate_percent']:.2f}%")
    print(f"Truncated Rate: {summary['truncated_rate_percent']:.2f}%")
    print(f"Best Return: {summary['best_return']:.3f}")
    print(f"Worst Return: {summary['worst_return']:.3f}")
    print(f"Best Final Distance: {summary['best_final_distance']:.3f}")
    print(f"Worst Final Distance: {summary['worst_final_distance']:.3f}")
    print("=====================\n")


def save_evaluation_summary(episode_stats: list[EpisodeStats], summary: dict[str, float|int], output_path: Path) -> None:
    """Saves the episode stats and summary to a JSON file."""

    output_path=Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "summary": summary,
        "episodes": [asdict(e) for e in episode_stats]
    
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=4)

    print(f"Saved evaluation summary to {output_path}")


def _get_distance(info: dict[str, Any], obs: np.ndarray) -> float:

    if "distance_to_target" in info:
        return float(info["distance_to_target"])
    else:
        return float(obs[3])













