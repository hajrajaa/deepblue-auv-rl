#!/usr/bin/env python3

from deepblue_auv_rl.envs import AUVTargetEnv, MissionConfig


def main() -> None:
    print("Running local Phase 2 check without starting HoloOcean...")

    config = MissionConfig()
    env = AUVTargetEnv(config=config, auto_start=False)

    print("Action space:", env.action_space)
    print("Observation space:", env.observation_space)

    assert env.action_space.n == 6
    assert env.observation_space.shape == (8,)

    print("Action meanings:")
    print("0 = forward")
    print("1 = turn left")
    print("2 = turn right")
    print("3 = move up")
    print("4 = move down")
    print("5 = stop/collect")

    print("\nOK: Local wrapper structure check passed.")
    print("Important: this did not start HoloOcean.")
    print("Run the real simulator tests later on RunPod.")


if __name__ == "__main__":
    main()