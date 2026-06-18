#!/usr/bin/env python3


import argparse



import numpy as np

from deepblue_auv_rl.envs import AUVTargetEnv, MissionConfig


def parse_args()-> argparse.Namespace:
    parser=argparse.ArgumentParser(description="Run RANDOM POLICY ROLLOUTS auv TARGET MISSION ")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--viewport",action="store_true")
    return parser.parse_args()

def main()->None:

    args=parse_args()

    config=MissionConfig(max_steps=args.steps)
    env=AUVTargetEnv(config=config,show_viewport=args.show_view)

    episode_returns:list[float]=[]
    final_distances:list[float]=[]
    successes:list[bool]=[]

    try:

        for episode in range(args.episodes):
            obs,info=env.reset(seed=args.seed + episode)

            total_reward=0.0

            min_distance=info["distance_to_target"]

            for step in range(args.steps):
                action=env.action_space.sample()
                obs, reward, terminated, truncated, info=env.step(action)
                total_reward+=reward

                if terminated or truncated:
                    break
            
            episode_returns.append(total_reward)
            final_distances.append(info["distance_to_target"])
            successes.append(info["success"])

            print(
                f"episode={episode + 1:03d} "
                f"steps={info['step']:03d} "
                f"return={total_reward:9.3f} "
                f"final_dist={info['distance_to_target']:8.3f} "
                f"min_dist={min_distance:8.3f} "
                f"success={info['success']} "
                f"out_of_bounds={info['out_of_bounds']}"
            )

        print("\n=== Random Rollout Metrics ===")
        print(f"Episodes: {args.episodes}")
        print(f"Mean return: {np.mean(episode_returns):.3f}")
        print(f"Mean final distance: {np.mean(final_distances):.3f}")
        print(f"Success rate: {100.0 * np.mean(successes):.1f}%")


    finally:
        env.close()

if __name__ == "__main__":
    main()
    




