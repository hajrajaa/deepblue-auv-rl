#!/usr/bin/env python3

from deepblue_auv_rl.envs import AUVTargetEnv, MissionConfig


def main()->None:

    config=MissionConfig(max_steps=20)
    env=AUVTargetEnv(config=config,show_viewport=False,varebose=True)

    try:

        obs,info=env.reset(seed=0)

        print("\nInitial observation:", obs)

        print("\nInitial info:", info)

        print("\nHoloOcean state keys:")
        print(env.debug_state_keys())

        print("\nTaking one Forward action...")

        obs, reward, terminated, truncated, info=env.step(0)

        print("\nObservation after action:", obs)

        print("\nReward:", reward)

        print("\nTerminated:", terminated)
        print("\nTruncated:", truncated)

        print("\nInfo after action:", info)
        
        print("\nHoloOcean state keys after action:")
        print(env.debug_state_keys())

    finally:
        env.close()


if __name__ == "__main__":
    main()
    







