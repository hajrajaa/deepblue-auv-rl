#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from deepblue_auv_rl.evaluation import (
    evaluate_policy,
    print_evaluation_summary,
    random_policy,
    save_evaluation_summary,
    summarize_episode_stats,
)


class TinyLocalAUVEnv(gym.Env):
    """
    Tiny local environment for testing metrics without HoloOcean.

    This is not the final simulator.
    It is only for checking evaluation code locally.
    """

    def __init__(self) -> None:
        super().__init__()

        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(
            low=-1000.0,
            high=1000.0,
            shape=(7,),
            dtype=np.float32,
        )

        self.position = np.array([0.0, 0.0, -5.0], dtype=np.float32)
        self.target = np.array([5.0, 0.0, -5.0], dtype=np.float32)
        self.step_count = 0
        self.max_steps = 30
        self.reach_threshold = 0.5
        self.previous_distance = self._distance()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        super().reset(seed=seed)

        self.position = np.array([0.0, 0.0, -5.0], dtype=np.float32)
        self.target = np.array([5.0, 0.0, -5.0], dtype=np.float32)
        self.step_count = 0
        self.previous_distance = self._distance()

        obs = self._obs()
        info = {
            "distance_to_target": self.previous_distance,
            "success": False,
            "out_of_bounds": False,
        }

        return obs, info

    def step(self, action: int):
        self.step_count += 1

        if action == 0:
            self.position[0] += 1.0
        elif action == 1:
            self.position[1] += 0.5
        elif action == 2:
            self.position[1] -= 0.5
        elif action == 3:
            self.position[2] += 0.5
        elif action == 4:
            self.position[2] -= 0.5
        elif action == 5:
            pass

        current_distance = self._distance()
        progress = self.previous_distance - current_distance

        reward = -1.0 + 5.0 * progress

        success = current_distance <= self.reach_threshold
        out_of_bounds = bool(np.any(np.abs(self.position) > 20.0))

        terminated = success or out_of_bounds
        truncated = self.step_count >= self.max_steps and not terminated

        if success:
            reward += 100.0

        if out_of_bounds:
            reward -= 50.0

        self.previous_distance = current_distance

        obs = self._obs()
        info = {
            "distance_to_target": current_distance,
            "success": success,
            "out_of_bounds": out_of_bounds,
            "step": self.step_count,
        }

        return obs, float(reward), terminated, truncated, info

    def _distance(self) -> float:
        return float(np.linalg.norm(self.position - self.target))

    def _obs(self) -> np.ndarray:
        return np.array(
            [
                self.position[0],
                self.position[1],
                self.position[2],
                self.target[0],
                self.target[1],
                self.target[2],
                self._distance(),
            ],
            dtype=np.float32,
        )


def greedy_forward_policy(obs: np.ndarray, info: dict[str, Any], env: Any) -> int:
    """
    Simple hand-coded policy for testing.
    Since the target is at x=5, always move forward.
    """
    return 0


def main() -> None:
    env = TinyLocalAUVEnv()

    print("\nTesting random policy metrics...")
    random_stats = evaluate_policy(
        env=env,
        policy_fn=random_policy,
        n_episodes=5,
        max_steps=30,
        seed=0,
        verbose=True,
    )

    random_summary = summarize_episode_stats(random_stats)
    print_evaluation_summary(random_summary, title="Random Policy - Local Test")

    save_evaluation_summary(
        episode_stats=random_stats,
        summary=random_summary,
        output_path=Path("results/evaluation/random_policy_local_test.json"),
    )

    print("\nTesting greedy forward policy metrics...")
    greedy_stats = evaluate_policy(
        env=env,
        policy_fn=greedy_forward_policy,
        n_episodes=5,
        max_steps=30,
        seed=0,
        verbose=True,
    )

    greedy_summary = summarize_episode_stats(greedy_stats)
    print_evaluation_summary(greedy_summary, title="Greedy Forward Policy - Local Test")

    save_evaluation_summary(
        episode_stats=greedy_stats,
        summary=greedy_summary,
        output_path=Path("results/evaluation/greedy_policy_local_test.json"),
    )

    print("\nOK: Evaluation metrics test finished.")


if __name__ == "__main__":
    main()