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

def make_env(seed:int):
    def _init():
        config=MissionConfig(
            random_start=False,
            random_target=False,)
        env= AUVTargetEnv(
            config=config,
            show_viewport=False,
            verbose=False,
            auto_start=False,
            )
        env= Monitor(env, filename=str(LOG_DIR / "train_monitor.csv"))
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

    return parser.parse_args()

def main():

    args=parse_args()
    
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    set_random_seed(args.seed)

    print("=" * 50)
    print(f"Training PPO for AUV Target Environment")
    print("=" * 50)
    print(f"Total Timesteps: {args.total_timesteps}")
    print(f"Seed: {args.seed}")
    print(f"Device: {args.device}")
    print(f"Model directory: {MODEL_DIR}")
    print(f"Log directory: {LOG_DIR}")
    print("=" * 50)


    env= DummyVecEnv([make_env(args.seed)])

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

    model_path= MODEL_DIR / f"{args.model_name}_{args.total_timesteps}_steps.zip"
    model.save(str(model_path))

    summary={
        "algorithm": "PPO",
        "policy": "MlpPolicy",
        "total_timesteps": args.total_timesteps,
        "seed": args.seed,
        "device": args.device,
        "model_path": str(model_path),
        "elapsed_time_seconds": elapsed_time,
        "environment": {
            "fixed_target": True,
            "fixed_target_position": True,
            "obstacles": False,
            "random_start": False,
            "random_target": False,

        },
    }

    summary_path= MODEL_DIR / f"{args.model_name}_{args.total_timesteps}_training_summary.json"

    with summary_path.open("w",encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    env.close()

    print("\nTraining completed!")
    print(f"Model saved to: {model_path}")
    print(f"Training summary saved to: {summary_path}")
    print(f"Monitor logs saved to: {LOG_DIR/'train_monitor.csv'}")


if __name__=="__main__":
    main()


       


