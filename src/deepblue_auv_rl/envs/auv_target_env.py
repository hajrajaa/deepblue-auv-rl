from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

ACTION_NAMES={
    0:"forward",
    1:"turn_left",
    2:"turn_right",
    3:"move_up",
    4:"move_down",
    5:"stop_collect",
}


@dataclass
class MissionConfig:


    world:str="PierHarbor"
    agent_name:str="auv0"


    #fixed simple start and target for now 
    start_position:tuple[float,float,float]=(0.0,0.0,-5.0)
    target_position:tuple[float,float,float]=(10.0,0.0,-5.0)

    random_start:bool=False
    random_target:bool=False

    #Mission limits. If the AUV leaves this box , the episode will terminate.
    bounds_min:tuple[float,float,float]=(-50.0,-50.0,-30.0)
    bounds_max:tuple[float,float,float]=(50.0,50.0,0.0)

    #Eposide settings
    max_steps:int=200
    reach_threshold:float=1.0

    #Discrete action sizes 
    forward_step:float=1.0
    vertical_step:float=0.75
    turn_degrees:float=15.0

    # HoloOcean simulation ticks per one Gymnasium step.
    ticks_per_action:int=5

    #Reward parameters
    goal_reward:float=200.0
    step_penalty:float=-0.05
    progress_reward_scale:float=10.0
    out_of_bounds_penalty:float=-50.0

    moving_target:bool=False
    target_velocity:tuple[float,float,float]=(0.03,0.02,0.0)

    obstacles_enabled:bool=False
    moving_obstacles:bool=False
    num_obstacles:int=0
    obstacle_radius:float=1.0
    safe_distance:float=2.0
    max_obstacle_sensor_range:float=20.0

    collision_penalty:float=-100.0
    obstacle_penalty_scale: float = 5.0

    obstacle_velocity:tuple[float,float,float]=(0.0, 0.02, 0.0)

    

    # obstacle_positions:tuple[tuple[float,float,float],...]=(
    #     (3.0, 0.0, -5.0),
    # )
    obstacle_positions: tuple[tuple[float, float, float], ...] = (
    (5.0, 2.0, -5.0),
    )





class AUVTargetEnv(gym.Env):

    metadata={"render_modes":[]}

    def __init__(self, 
                 config:MissionConfig | None=None,
                 show_viewport:bool=False,
                 verbose:bool=False,
                 render_mode:str|None=None,
                 auto_start:bool=True,
    )-> None:
        super().__init__()

        self.config=config or MissionConfig()
        self.show_viewport=show_viewport
        self.verbose=verbose
        self.render_mode=render_mode

        self.action_space=spaces.Discrete(6)

        # # change this when start trainig 
        # obs_low=np.array( [-1000.0, -1000.0, -1000.0, -1000.0, -1000.0, -1000.0, 0.0], dtype=np.float32)
        # obs_high=np.array([1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1.0], dtype=np.float32)

        self.observation_space = spaces.Box(
            low=np.array([
                -50.0, -50.0, -50.0, 0.0,   # target dx, dy, dz, distance
                -50.0, -50.0, -50.0, 0.0,   # obstacle dx, dy, dz, distance
                -1.0, -1.0,                 # sin(yaw), cos(yaw)
            ], dtype=np.float32),
            high=np.array([
                50.0, 50.0, 50.0, 100.0,
                50.0, 50.0, 50.0, self.config.max_obstacle_sensor_range,
                1.0, 1.0,
            ], dtype=np.float32),
            dtype=np.float32,
        )

        self._holo_env:Any|None=None
        self._last_raw_state:Any|None=None
        self._obstacle_positions_runtime:np.ndarray|None=None

        self.current_step=0

        self.commanded_position=np.array(self.config.start_position, dtype=np.float32)
        self.target_position=np.array(self.config.target_position, dtype=np.float32)
        self.yaw_deg=0.0
        self.previous_distance:float| None=None

        if auto_start:
            self._ensure_holoocean_env()


    def _build_scenario_cfg(self)-> dict[str, Any]:

        return{
            "name":"TargetMission",
            "package_name":"Ocean",
            "world":self.config.world,
            "main_agent":self.config.agent_name,

            "ticks_per_sec":30,
            "frames_per_sec":False,
            "env_min":list(self.config.bounds_min),
            "env_max":list(self.config.bounds_max),
            "window_width":1280,
            "window_height":720,
            "agents":[
                {
                    "agent_name":self.config.agent_name,
                    "agent_type":"HoveringAUV",
                    "control_scheme":1,
                    "location":list(self.config.start_position),
                    "rotation":[0.0, 0.0, 0.0],
                    "sensors":[
                        {
                            "sensor_type":"LocationSensor",
                            "sensor_name":"LocationSensor",
                            "Hz":30,
                            "configure": {
                                "Sigma":0.0,
                            },
                        },
                        {
                            "sensor_type":"RotationSensor",
                            "sensor_name":"RotationSensor",
                            "Hz":30
                        },
                        {
                            "sensor_type":"PoseSensor",
                            "sensor_name":"PoseSensor",
                            "Hz":30
                        },
                    ],
                }
            
            ],
        }
    
    def _ensure_holoocean_env(self)-> None:
        if self._holo_env is not None:
            return 

        try:
            import holoocean 
        except Exception as e:
            print("Failed to import HoloOcean:", repr(e))
            raise RuntimeError("HoloOcean is required to run this environment. Please install it and try again.") from e

        scenario_cfg=self._build_scenario_cfg()

        try:
            self._holo_env=holoocean.make(
                scenario_cfg=scenario_cfg,
                show_viewport=self.show_viewport,
                verbose=self.verbose,
            )
        except TypeError:
            try:
                self._holo_env=holoocean.make(
                scenario_cfg=scenario_cfg,
                show_viewport=self.show_viewport,
                
            )
            except TypeError:
                self._holo_env=holoocean.make(
                scenario_cfg=scenario_cfg
            )
                
        if hasattr(self._holo_env, "set_render_quality"):
            try:
                self._holo_env.set_render_quality(0)
            except Exception :
                pass
        if hasattr(self._holo_env, "set_control_scheme"):
            try:
                self._holo_env.set_control_scheme(self.config.agent_name,1)
            except Exception :
                pass


    def reset(self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
            ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._ensure_holoocean_env()
        self.current_step=0
        self.yaw_deg=0.0

        self.commanded_position = self.__choose_start_position(options)
        self.target_position = self.__choose_target_position(options)
        self._initialize_obstacle_positions()

        raw_state=self._holo_env.reset()

        teleport_ok=False

        if self.config.random_start or options is not None:
            teleport_ok=self._try_teleport_agent(self.commanded_position)

            raw_state=self._holoocean_step(self._make_pd_action(),ticks=1)

        self._last_raw_state=raw_state

        position=self._extract_position(raw_state)
        self.commanded_position=position.astype(np.float32).copy()

        distance=self._distance_to_target(position)
        self.previous_distance=distance

        observation=self._make_observation(position)
        info=self._make_info(
            position=position,
            distance=distance,
            action=None,
            reward=0.0,
            terminated=False,
            truncated=False,
            success=False,
            out_of_bounds=False,
            collision=False,
        )
        info["teleport_ok"]=teleport_ok

        return observation, info


    def step(self, action:int)-> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:

        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        
        self.current_step+=1

        if self.previous_distance is None:
            raise RuntimeError("Environment must be reset before stepping.")

        
        # Optional: move target if moving_target is enabled
        if self.config.moving_target:
            self.target_position=(self.target_position+np.asarray(self.config.target_velocity, dtype=np.float32)).astype(np.float32)

        if self.config.moving_obstacles:
            self._advance_obstacles()

        
        
        # apply agent action 
        self._apply_discrete_action(int(action))

        raw_state=self._holoocean_step(self._make_pd_action(), ticks=self.config.ticks_per_action)

        self._last_raw_state=raw_state

        position=self._extract_position(raw_state)
        distance=self._distance_to_target(position)

        # get optional info about closest obstacle
        _,closest_obstacle_distance=self._get_closest_obstacle(position)

        collision=(
            self.config.obstacles_enabled
            and closest_obstacle_distance<=self.config.obstacle_radius
        )

        # Main target reaching reward 
        progress=self.previous_distance-distance
        reward=self.config.progress_reward_scale*progress + self.config.step_penalty

        #Obstcale safe distance penalty
        if self.config.obstacles_enabled and closest_obstacle_distance<=self.config.safe_distance:
            reward-=self.config.obstacle_penalty_scale*(self.config.safe_distance-closest_obstacle_distance)

        success=distance<=self.config.reach_threshold
        out_of_bounds=self._is_out_of_bounds(position)

        terminated=False
        truncated=False

        if success:

            reward+=self.config.goal_reward
            terminated=True

        if out_of_bounds:

            reward+=self.config.out_of_bounds_penalty
            terminated=True

        if collision:
            
            reward+=self.config.collision_penalty
            terminated=True


        if self.current_step>=self.config.max_steps and not terminated:
            truncated=True

        self.previous_distance=distance

        observation=self._make_observation(position)
        info=self._make_info(
            position=position,
            distance=distance,
            action=int(action),
            reward=float(reward),
            terminated=terminated,
            truncated=truncated,
            success=success,
            out_of_bounds=out_of_bounds,
            collision=collision,
        )

        return observation, float(reward), terminated, truncated, info

    def __choose_start_position(self,options:dict[str, Any]|None)-> np.ndarray:

        if options and "start_position" in options:
            return np.array(options["start_position"], dtype=np.float32)
        
        if not self.config.random_start:
            return np.array(self.config.start_position, dtype=np.float32)
        
        low = np.array([-2.0, -2.0, -6.0], dtype=np.float32)
        high = np.array([2.0, 2.0, -4.0], dtype=np.float32)
        return self.np_random.uniform(low=low, high=high).astype(np.float32)

    def _initialize_obstacle_positions(self)-> None:
        if not self.config.obstacles_enabled or self.config.num_obstacles == 0:
            self._obstacle_positions_runtime = None
            return

        self._obstacle_positions_runtime = np.asarray(
            self.config.obstacle_positions[: self.config.num_obstacles],
            dtype=np.float32,
        ).copy()

    def _advance_obstacles(self)-> None:
        if self._obstacle_positions_runtime is None:
            return

        velocity = np.asarray(self.config.obstacle_velocity, dtype=np.float32)
        self._obstacle_positions_runtime = (
            self._obstacle_positions_runtime + velocity
        ).astype(np.float32)

    def __choose_target_position(self,options:dict[str, Any]|None)-> np.ndarray:
        if options and "target_position" in options:
            return np.array(options["target_position"], dtype=np.float32)
        
        if not self.config.random_target:
            return np.array(self.config.target_position, dtype=np.float32)
        
        low=np.array([-5.0, -8.0, -8.0], dtype=np.float32)
        high=np.array([15.0, 8.0, -3.0], dtype=np.float32)
        return self.np_random.uniform(low=low, high=high).astype(np.float32)

    def _apply_discrete_action(self, action:int)-> None:

        yaw_rad=math.radians(self.yaw_deg)

        if action==0:
            self.commanded_position[0]+=self.config.forward_step*math.cos(yaw_rad)
            self.commanded_position[1]+=self.config.forward_step*math.sin(yaw_rad)

        elif action==1:
            self.yaw_deg+=self.config.turn_degrees

        elif action==2:
            self.yaw_deg-=self.config.turn_degrees

        elif action==3:
            self.commanded_position[2]+=self.config.vertical_step

        elif action==4:
            self.commanded_position[2]-=self.config.vertical_step

        elif action==5:
            pass
    def _make_pd_action(self)-> np.ndarray:

        return np.array(
            [
                self.commanded_position[0],
                self.commanded_position[1],
                self.commanded_position[2],
                0.0,
                0.0,
                self.yaw_deg,
            ]
        , dtype=np.float32
        )

    def _holoocean_step(self, pd_action:np.ndarray, ticks:int)-> Any:

        ticks= max(1, int(ticks))

        pd_action=np.asarray(pd_action, dtype=np.float32)

        try:

            return self._holo_env.step(pd_action, ticks=ticks)
        except TypeError:
            pass

        state=None
        for _ in range(ticks):
            try:
                state=self._holo_env.step(pd_action)
            except TypeError:
                self._holo_env.act(self.config.agent_name, pd_action)
                state=self._holo_env.tick()

        return state


    def _try_teleport_agent(self, position:np.ndarray)-> bool:

        if self._holo_env is None:
            return False
        
        rotation=np.array([0.0, 0.0, self.yaw_deg], dtype=np.float32)

        try:
            agents=getattr(self._holo_env, "agents", None)

            agent=None

            if isinstance(agents, dict):
                agent=agents.get(self.config.agent_name, None)
            elif agents is not None :
                try:
                    agent=agents[self.config.agent_name]

                except Exception:
                    agent=None

            if agent is not None and hasattr(agent, "teleport"):
                agent.teleport(location=np.asarray(position), rotation=rotation)
                return True
        except Exception:
            return False
        
        return False

    def _state_for_agent(self, raw_state:Any)-> dict[str, Any]|None:
        if not isinstance(raw_state, dict):
            raise TypeError(f"Expected HoloOcean state to be a dict, got {type(raw_state)}")
        
        if self.config.agent_name  in raw_state and isinstance(raw_state[self.config.agent_name], dict):
            return raw_state[self.config.agent_name]
        
        return raw_state 

    def _extract_position(self, raw_state:Any)-> np.ndarray:

        sensors=self._state_for_agent(raw_state)

        for key in ["LocationSensor", "location","Location"]:
            if key in sensors :
                position=np.asanyarray(sensors[key], dtype=np.float32).reshape(-1)

                if position.size>=3:
                    return position[:3].astype(np.float32)
                
        if "PoseSensor" in sensors:

            pose=np.asanyarray(sensors["PoseSensor"], dtype=np.float32)

            if pose.shape[0]>=3 and pose.shape[1]>=4:
                return pose[:3,3].astype(np.float32)
            
        if "DynamicSensor" in sensors:
            dynamics=np.asanyarray(sensors["DynamicSensor"], dtype=np.float32).reshape(-1)

            if dynamics.size>=9:
                return dynamics[6:9].astype(np.float32)

        
        available=", ".join(str(k) for k in sensors.keys())

        raise KeyError(
            f"Could not find position in HoloOcean state. Available keys: {available}"
        )

    def _make_observation(self, position: np.ndarray) -> np.ndarray:
        position = position.astype(np.float32)

        target_delta = self.target_position.astype(np.float32) - position
        distance_to_target = float(np.linalg.norm(target_delta))

        obstacle_delta, closest_obstacle_distance = self._get_closest_obstacle(position)

        yaw_rad = math.radians(self.yaw_deg)
        sin_yaw = math.sin(yaw_rad)
        cos_yaw = math.cos(yaw_rad)

        observation = np.array([
            target_delta[0], target_delta[1], target_delta[2], distance_to_target,
            obstacle_delta[0], obstacle_delta[1], obstacle_delta[2], closest_obstacle_distance,
            sin_yaw, cos_yaw,
        ], dtype=np.float32)

        return observation
    
    def _get_closest_obstacle(self, position:np.ndarray)-> tuple[np.ndarray, float]:

        if not self.config.obstacles_enabled or self.config.num_obstacles==0:
            dummy_delta=np.array([0.0, 0.0, 0.0], dtype=np.float32)
            return dummy_delta, float(self.config.max_obstacle_sensor_range)
        
        if self._obstacle_positions_runtime is not None:
            obstacle_positions=self._obstacle_positions_runtime
        else:
            obstacle_positions=np.asarray(self.config.obstacle_positions[:self.config.num_obstacles], dtype=np.float32)

        deltas=obstacle_positions-position.astype(np.float32)
        distances=np.linalg.norm(deltas, axis=1)

        closest_idx=int(np.argmin(distances))
        closest_delta=deltas[closest_idx].astype(np.float32)
        closest_distance=float(distances[closest_idx])

        return closest_delta, closest_distance



    def _distance_to_target(self, position:np.ndarray)-> float:
        
        return float(np.linalg.norm(position.astype(np.float32)-self.target_position))

    def _is_out_of_bounds(self, position:np.ndarray)-> bool:

        bounds_min=np.asarray(self.config.bounds_min, dtype=np.float32)
        bounds_max=np.asarray(self.config.bounds_max, dtype=np.float32)
        return bool(np.any(position<bounds_min) or np.any(position>bounds_max))

    def _make_info(self,
                *,
                position:np.ndarray,
                distance:float,
                action:int|None,
                reward:float,
                terminated:bool,
                truncated:bool,
                success:bool,
                out_of_bounds:bool ,
                collision:bool=False,
                )-> dict[str, Any]:
        _, closest_obstacle_distance=self._get_closest_obstacle(position)
        return {
            "step": self.current_step,
            "position": position.astype(np.float32).tolist(),
            "target_position": self.target_position.astype(np.float32).tolist(),
            "distance_to_target": float(distance),
            "closest_obstacle_distance": float(closest_obstacle_distance),
            "safe_distance": float(self.config.safe_distance),
            "collision": bool(collision),
            "previous_distance": None if self.previous_distance is None else float(self.previous_distance),
            "action": action,
            "action_name": None if action is None else ACTION_NAMES.get(action, "unknown"),
            "reward": float(reward),
            "success": bool(success),
            "out_of_bounds": bool(out_of_bounds),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
           
        }

    def debug_state_keys(self)->dict[str, Any]:
        if self._last_raw_state is None:
            raise RuntimeError("No state available. Please reset the environment first.")
        
        if isinstance(self._last_raw_state, dict):
            agent_state=self._state_for_agent(self._last_raw_state)
            return {
                "raw_top_level_keys": list(self._last_raw_state.keys()),
                "agent_sensor_keys":list(agent_state.keys()),}
        return {"raw_state_type": str(type(self._last_raw_state))}

    def render(self)->None:
        return None

    def close(self)->None:
        if self._holo_env is  None:
            return
        
        try:
            if hasattr(self._holo_env, "close"):
                self._holo_env.close()
            elif hasattr(self._holo_env, "__exit__"):
                self._holo_env.__exit__(None, None, None)

        except Exception:
            pass

        finally:

            self._holo_env=None





        
    



        























