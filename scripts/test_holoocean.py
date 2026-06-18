import argparse
import sys
import time 

import numpy as np



CANDIDATE_SCENARIOS = [
    "PierHarbor-Hovering",
    "SimpleUnderwater-AUV",
    "SimpleUnderwater-Hovering",
    "OpenWater-Hovering",
    "Dam-Hovering",
]


def get_zero_action(env):

    action_space=getattr(env, "action_space", None)

    if action_space is not None:
        print("\nAction space:", action_space)

        if hasattr(action_space, "sample"):

            try:
                sample=action_space.sample()
                return np.zeros_like(sample,dtype=np.float32)
            except Exception as e:
                print("Could not sample action space:", repr(e))
                

        shape=getattr(action_space, "shape", None)
        if shape is not None:
            return np.zeros(shape,dtype=np.float32)
        
    print("Could not determine action space shape, defaulting to 8-dimensional zero action.")
    return np.zeros(8,dtype=np.float32)  # Default to 8-dimensional zero action


def summarize_state(state):

    print("\nState type:", type(state))

    if isinstance(state, dict):
        print("Top-level state keys:", list(state.keys()))

        for key, value in state.items():
            print(f"\nAgent/key: {key}")

            if isinstance(value, dict):
                print(" Sensor keys:", list(value.keys()))
                for sensor_name, sensor_value in value.items():
                    try:
                        arr=np.asanyarray(sensor_value)
                        print(f" {sensor_name}: shape:={arr.shape } dtype:{arr.dtype}")

                    except Exception:
                        print(f"   {sensor_name}:{type(sensor_value)}")

            else:
                try:
                    arr=np.asanyarray(value)
                    print(f" Value shape:={arr.shape } dtype:{arr.dtype}")

                except Exception:
                    print(f"   Value:{type(value)}")

    else:
        try:
            arr=np.asanyarray(state)
            print(f" State shape:={arr.shape } dtype:{arr.dtype}")

        except Exception:
            print(f"   State:{type(state)}")



def scenario_exists(packagemanager, scenario_name):
    try:
        packagemanager.get_scenario(scenario_name)
        return True
    except Exception :
        return False
    

def choose_scenario(packagemanager, requested):

    if requested:
        return requested
    
    for scenario in CANDIDATE_SCENARIOS:
        if scenario_exists(packagemanager, scenario):
            return scenario
        
    print("Could not find one of the candidate scenarios.")
    print("Run this command to inspect available scenarios:")
    print("  python scripts/check_holoocean_install.py")
    sys.exit(1)

def main():

    parser=argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default=None)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--viewport",action="store_true")
    args=parser.parse_args()

    try:
        import holoocean 
        from holoocean import packagemanager

    except Exception as e:
        print("Failed to import HoloOcean.")
        print("Error:", repr(e))
        sys.exit(1)

    scenario_name=choose_scenario(packagemanager, args.scenario)

    print("Using scenario:", scenario_name)
    print("show_viewport:", args.viewport)
    
    env=None 

    try:

        env=holoocean.make(
            scenario_name,
            show_viewport=args.viewport,
            verbose=True,
            ticks_per_second=30,
            frames_per_sec=False,
        )

        try:
            env.set_render_quality(0)
        except Exception as e:
            print("Could not set render quality:", repr(e))

        try:
            print("\nEnviroment info:")
            print(env.info())

        except Exception as e:
            print("Could not get environment info:", repr(e))

        print("\nResetting environment...")
        state=env.reset()
        summarize_state(state)

        action=get_zero_action(env)
        print("\nUsing zero action:", action)

        print("\nRunning {args.steps} steps in the environment...")

        start_time=time.time()

        for step in range(args.steps):
            state=env.step(action)

            if step in [0, args.steps-1]:
                print(f"\nStep {step+1}/{args.steps}")
                summarize_state(state)

        elapsed_time=time.time()-start_time
        print(f"\nSUCCESS: HoloOcean ran{args.steps} steps in {elapsed_time:.2f} seconds.")

    except Exception as e:
        print("\nFAILED while running HoloOcean:", repr(e))
        print("\nTry running with xvfb:")
        print("xvfb-run -s '-screen 0 1280x720x24' python scripts/test_holoocean.py")
        sys.exit(1)

    finally:
        if env is not None and hasattr(env, "close"):
            try:
                env.close()
            except Exception as e:
                print("Could not close environment:", repr(e))

        else:
            print("Environment does not have a close() method, skipping cleanup.")


if __name__ == "__main__":
    main()
    





