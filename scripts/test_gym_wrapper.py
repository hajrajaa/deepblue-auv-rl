#!/usr/bin/env python3

import argparse

from deepblue_auv_rl.envs import AUVTargetEnv, MissionConfig

def parse_args()-> argparse.Namespace:

    parser=argparse.ArgumentParser(description="Test the Gym wrapper for HoloOcean scenarios.")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--viewport",action="store_true")
    parser.add_argument("--check-env" , action="store_true", help="Run the environment check utility to verify the installation.")
    return parser.parse_args()

def main()->None: 
    args=parse_args()

    config=MissionConfig(max_steps=args.steps)
    env=AUVTargetEnv(config=config,show_viewport=args.viewport)

    try:
        if args.check_env:
            print("Running environment check utility...")
            from stable_baselines3.common.env_checker import check_env

            check_env(env, warn=True,skip_render_check=True)
            print("Environment check passed successfully!")

        obs,info=env.reset(seed=args.seed)

        print("\nInitial observation:", obs)
        print("\nInitial info:", info)

        print("\nStrating random-paction wrapper test...\n")

        total_reward=0.0

        for step in range(args.steps):
            action=env.action_space.sample()
            obs, reward, terminated, truncated, info=env.step(action)
            total_reward+=reward

            print(
                f"step={step + 1:03d} "
                f"action={action}({info['action_name']}) "
                f"reward={reward:8.3f} "
                f"dist={info['distance_to_target']:8.3f} "
                f"success={info['success']} "
                f"terminated={terminated} "
                f"truncated={truncated}"
            )

            if terminated or truncated:
                break
        print(
                f"step={step + 1:03d} "
                f"action={action}({info['action_name']}) "
                f"reward={reward:8.3f} "
                f"dist={info['distance_to_target']:8.3f} "
                f"success={info['success']} "
                f"terminated={terminated} "
                f"truncated={truncated}"
            )
        
    finally:
        env.close()

if __name__ == "__main__":
    main()
           







