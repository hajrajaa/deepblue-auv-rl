#!/usr/bin/env python3

from __future__ import annotations
import argparse
import json
import time 
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed

from deepblue_auv_rl.envs.auv_target_env import AUVTargetEnv,MissionConfig


REPO_ROOT= Path(__file__).resolve().parents[1]
MODEL_DIR= REPO_ROOT / "models"
LOG_DIR= REPO_ROOT /"logs"
TENSORBOARD_DIR= REPO_ROOT /"logs" / "tensorboard"
RESULT_DIR= REPO_ROOT / "results"/"evaluation"


def build_env_config(stage:str, max_steps:int)-> MissionConfig:
    """ Build a MissionConfig based on the training stage."""

    if stage=="fixed_no_obstacles":
        return MissionConfig(
            random_start=False,
            random_target=False,
            moving_target=False,
            obstacles_enabled=False,
            max_steps=max_steps,
        )
    if stage=="random_start_no_obstacles":
        return MissionConfig(
            random_start=True,
            random_target=False,
            moving_target=False,
            obstacles_enabled=False,
            max_steps=max_steps,
        )
    if stage=="moving_no_obstacles":
        return MissionConfig(
            random_start=True,
            random_target=False,
            moving_target=True,
            obstacles_enabled=False,
            max_steps=max_steps,
        )
    if stage=="fixed_obstacles":
        return MissionConfig(
            random_start=False,
            random_target=False,
            moving_target=False,
            obstacles_enabled=True,
            num_obstacles=1,
            max_steps=max_steps,
        )
    if stage=="moving_obstacles":
        return MissionConfig(
            random_start=True,
            random_target=False,
            moving_target=False,
            obstacles_enabled=True,
            num_obstacles=1,
            moving_obstacles=True,
            max_steps=max_steps,
        )
    raise ValueError(f"Unknown training stage: {stage}. Valid stages are: fixed_no_obstacles, random_start_no_obstacles, moving_no_obstacles, fixed_obstacles, moving_obstacles.")
    
    

def make_env(seed:int,stage:str,monitor_path:Path,max_steps:int):
    def _init():
        config=build_env_config(stage,max_steps)
        env= AUVTargetEnv(
            config=config,
            show_viewport=False,
            verbose=False,
            auto_start=False,
            )
        env= Monitor(env, filename=str(monitor_path))
        env.reset(seed=seed)
        return env
    
    return _init

def parse_args():

    parser=argparse.ArgumentParser()
    parser.add_argument(
    "--total-timesteps",
    type=int,
    default=1000,
    help="Use 1000 for smoke test , then 20000 or more.")

    parser.add_argument(
    "--seed",
    type=int,
    default=42,
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="ppo_auv_target",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto","cpu","cuda"],
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="fixed_no_obstacles",
        choices=["fixed_no_obstacles","random_start_no_obstacles","moving_no_obstacles","fixed_obstacles","moving_obstacles"],
        help="Training stage for curriculum learning. Determines the environment configuration.",
    )

    parser.add_argument(
        "--load-model",
        type=str,
        default=None,
        help="Path to an existing PPO model to continue training from."
    )


    return parser.parse_args()

def main():

    args=parse_args()
    
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    set_random_seed(args.seed)

    run_name=f"{args.model_name}_{args.stage}_{args.total_timesteps}_steps_seed_{args.seed}"
    run_log_dir= LOG_DIR / run_name
    run_log_dir.mkdir(parents=True, exist_ok=True)
    monitor_path= run_log_dir / "train_monitor.csv"

    print("=" * 50)
    print(f"Training PPO for AUV Target Environment")
    print("=" * 50)
    print(f"Total Timesteps: {args.total_timesteps}")
    print(f"Seed: {args.seed}")
    print(f"Device: {args.device}")
    print(f"Max steps per episode: {args.max_steps}")
    print(f"Model directory: {MODEL_DIR}")
    print(f"Log directory: {LOG_DIR}")
    print("=" * 50)


    env= DummyVecEnv([make_env(args.seed, args.stage, monitor_path, args.max_steps)])


    if args.load_model is not None:
        print(f"Loading existing model from: {args.load_model}")
        model= PPO.load(args.load_model, env=env, device=args.device)
    else:
        print("Creating new PPO model.")
        model= PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=3e-4,
            n_steps=128,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            seed=args.seed,
            device=args.device,
            tensorboard_log=str(TENSORBOARD_DIR),
    )



    
    start_time= time.time()
    model.learn(total_timesteps=args.total_timesteps,
                tb_log_name=f"{args.model_name}_{args.total_timesteps}_steps",)
                
    elapsed_time= time.time() - start_time

    model_path= MODEL_DIR /f"{run_name}.zip"
    model.save(str(model_path))

    summary={
        "algorithm": "PPO",
        "policy": "MlpPolicy",
        "total_timesteps": args.total_timesteps,
        "seed": args.seed,
        "device": args.device,
        "max_steps": args.max_steps,
        "model_path": str(model_path),
        "elapsed_time_seconds": elapsed_time,
        "environment": {
            "stage": args.stage,
            "random_start": args.stage in ["random_start_no_obstacles","moving_no_obstacles","moving_obstacles"],
            "moving_target": args.stage == "moving_no_obstacles",
            "obstacles_enabled": args.stage in ["fixed_obstacles","moving_obstacles"],
            "moving_obstacles": args.stage == "moving_obstacles",
        },
    }

    summary_path= MODEL_DIR / f"{run_name}_training_summary.json"

    with summary_path.open("w",encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    env.close()

    print("\nTraining completed!")
    print(f"Model saved to: {model_path}")
    print(f"Training summary saved to: {summary_path}")
    


if __name__=="__main__":
    main()


       


