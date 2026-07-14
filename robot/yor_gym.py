import gymnasium as gym
import mujoco
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

from gymnasium import error, logger, spaces 
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import math
from typing import Any, Optional, Tuple, Union

import numpy as np

_HERE = Path(__file__).parent

def wrap_pi(a: np.ndarray) -> np.ndarray:
    return ((a + math.pi) % (2 * math.pi)) - math.pi


def diff_angle(a: np.ndarray, b: Union[np.ndarray, float]) -> np.ndarray:
    return ((a - b) + math.pi) % (2 * math.pi) - math.pi


def frac_to_rad(f: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return ((np.array(f) + 0.5) % 1.0 - 0.5) * TWO_PI


def rad_to_frac(rad: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return (np.array(rad) / TWO_PI) % 1.0

NUM_SWERVES = 4
LENGTH = 0.1225  # m
WIDTH = 0.170  # m
TIRE_RADIUS = 0.0381  # m

MODULE_ORDER = ("FL", "FR", "RR", "RL")

DRIVE_NAMES = ("drive_front_left_ctrl", "drive_front_right_ctrl", "drive_back_right_ctrl", "drive_back_left_ctrl")  # [FL, FR, RR, RL]
ROT_NAMES = ("front_left_steer_ctrl", "front_right_steer_ctrl", "back_right_steer_ctrl", "back_left_steer_ctrl")  # [FL, FR, RR, RL]

ROTATION_OFFSETS = np.array([0.75, 0.00, 0.25, 0.50], dtype=float)

ROT_DIAG_SWAP_PERM = np.array([1, 0, 3, 2], dtype=int)
TRANS_OPPOSITE_MASK = np.array([False, False, False, False], dtype=bool)

TWO_PI = 2.0 * math.pi

USE_FEEDBACK_FOR_STEER = False
DRIVE_VEL_SCALE = 2.0

class BaseMujoco():
    def __init__(
        self,
        mjcf_path: str, env_limit=10, 
        max_vel=np.array((1.0, 1.0, 1.57)),
        max_accel=np.array((1.0, 1.0, 1.57)),
    ):
        self.mjcf_path = mjcf_path

        self.model = mujoco.MjModel.from_xml_path(self.mjcf_path)
        self.data  = mujoco.MjData(self.model)

        self.initial_qpos = np.copy(self.data.qpos)
        self.initial_qvel = np.copy(self.data.qvel)

        self.dt = self.model.opt.timestep

        self.max_vel = max_vel
        self.max_accel = max_accel

        self.C = np.array(
            [
                [1, 0, WIDTH],
                [1, 0, -WIDTH],
                [1, 0, -WIDTH],
                [1, 0, WIDTH],
                [0, 1, LENGTH],
                [0, 1, LENGTH],
                [0, 1, -LENGTH],
                [0, 1, -LENGTH],
            ]
        )

        self.rotation_motors = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in ROT_NAMES]# [FL, FR, RR, RL]
        self.drive_motors = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in DRIVE_NAMES]# [FL, FR, RR, RL]


        self.steer_pos = np.zeros(NUM_SWERVES)
        self.drive_vel = np.zeros(NUM_SWERVES)
        self.x = np.zeros(3)
        self.dx = np.zeros(3)

        self.base_target = np.zeros(3)

        # --- S-curve profiling state (kept; now optional per-command) ---
        self._smooth_active = False  # whether to apply smoothing for the *current* command
        self._v_prof = np.zeros(3, dtype=float)
        self._seg_v0 = np.zeros(3, dtype=float)
        self._seg_v1 = np.zeros(3, dtype=float)
        self._seg_t = 0.0
        self._seg_T = 0.0

        self._a_max = np.array([1.9, 1.9, 6.5], dtype=float)
        self._T_min = 0.01
        self._retarget_eps = 1e-3

    def apply_action(self, cmd):

        self.base_target = np.array(cmd["target"], dtype=float)
        self._smooth_active = True # always true in physical

        for m in self.drive_motors:
            m.heartbeat()
        for m in self.rotation_motors:
            m.heartbeat()

        self._update_state()

        v_cmd = self.base_target

        if self._smooth_active:
            if np.linalg.norm(v_cmd - self._seg_v1) > self._retarget_eps:
                self._start_scurve_segment(v_cmd)
            v_used = self._update_scurve(self.dt)
        else:
            # Keep profiling state consistent so enabling smoothing later doesn't jump from stale state
            self._v_prof = v_cmd.copy()
            self._seg_v0 = v_cmd.copy()
            self._seg_v1 = v_cmd.copy()
            self._seg_t = 0.0
            self._seg_T = 0.0
            v_used = v_cmd

        wheel_speeds, wheel_angles = self._vehicle_velocity_to_angle_and_speed(
            v_used, cos_error_scaling=True
        )

        target_fracs = rad_to_frac(wheel_angles)
        for i, rm in enumerate(self.rotation_motors):
            self.data.ctrl[rm] = float(target_fracs[i])

        for i, dm in enumerate(self.drive_motors):
            self.data.ctrl[dm] = float(wheel_speeds[i])


    # -------------- helpers --------------
    def _update_state(self) -> None:

        for i, rm in enumerate(self.rotation_motors):
            self.steer_pos[i] = self.data.ctrl[rm]

        for i, dm in enumerate(self.drive_motors):
            self.drive_vel[i] = self.data.ctrl[dm]

    def _angle_and_speed_to_vehicle_velocity(
        self, wheel_speeds: np.ndarray, wheel_angles: np.ndarray
    ) -> np.ndarray:
        vx, vy = wheel_speeds * np.cos(wheel_angles), wheel_speeds * np.sin(wheel_angles)
        return np.linalg.lstsq(self.C, np.concatenate((vx, vy)), rcond=None)[0]

    def _start_scurve_segment(self, v_target: np.ndarray):
        v_target = np.asarray(v_target, dtype=float)

        if getattr(self, "_seg_T", 0.0) > 0 and np.allclose(v_target, self._seg_v1, atol=1e-3):
            return

        dv = v_target - self._v_prof
        abs_dv = np.abs(dv)

        if np.all(abs_dv < 1e-3):
            return

        T_needed = np.max((abs_dv * np.pi) / (2.0 * np.maximum(self._a_max, 1e-6)))
        T = max(self._T_min, float(T_needed))

        self._seg_v0 = self._v_prof.copy()
        self._seg_v1 = v_target.copy()
        self._seg_t = 0.0
        self._seg_T = T

    def _update_scurve(self, dt: float) -> np.ndarray:
        if self._seg_T <= 1e-9:
            return self._v_prof

        self._seg_t = min(self._seg_t + dt, self._seg_T)
        tau = self._seg_t / self._seg_T
        s = 0.5 * (1.0 - np.cos(np.pi * tau))
        self._v_prof = self._seg_v0 + (self._seg_v1 - self._seg_v0) * s
        return self._v_prof

    def _vehicle_velocity_to_angle_and_speed(
        self, u_3dof: np.ndarray, cos_error_scaling: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        vx, vy, omega = float(u_3dof[0]), float(u_3dof[1]), float(u_3dof[2])

        vx_t = np.array([vx, vx, vx, vx], dtype=float)
        vy_t = np.array([vy, vy, vy, vy], dtype=float)
        sign = np.where(TRANS_OPPOSITE_MASK, -1.0, 1.0)
        vx_t *= sign
        vy_t *= sign

        vx_r = np.array(
            [+WIDTH * omega, -WIDTH * omega, -WIDTH * omega, +WIDTH * omega], dtype=float
        )
        vy_r = np.array(
            [+LENGTH * omega, +LENGTH * omega, -LENGTH * omega, -LENGTH * omega],
            dtype=float,
        )
        vx_r = vx_r[ROT_DIAG_SWAP_PERM]
        vy_r = vy_r[ROT_DIAG_SWAP_PERM]

        vx_w = vx_t + vx_r
        vy_w = vy_t + vy_r

        wheel_speeds = np.hypot(vx_w, vy_w)
        wheel_angles = np.arctan2(vy_w, vx_w)

        error = diff_angle(wheel_angles, self.steer_pos)
        wheel_angles = np.where(
            np.abs(error) > np.pi / 2, diff_angle(wheel_angles, np.pi), wheel_angles
        )
        wheel_speeds = np.where(np.abs(error) > np.pi / 2, -wheel_speeds, wheel_speeds)

        if cos_error_scaling:
            wheel_speeds *= np.cos(diff_angle(wheel_angles, self.steer_pos))

        return wheel_speeds, wheel_angles

    def _map_steer_angles(self, wheel_angles: np.ndarray) -> np.ndarray:
        ang = wheel_angles.copy()
        ang[TRANS_OPPOSITE_MASK] = ang[TRANS_OPPOSITE_MASK] + math.pi
        ang = ang[ROT_DIAG_SWAP_PERM]
        return wrap_pi(ang)
    
    def reset(self):
        self.data.qpos = np.copy(self.initial_qpos)
        self.data.qvel = np.copy(self.initial_qvel)
        self.data.ctrl = np.zeros(self.model.nu)
        self.data.time = 0
    
        self.steer_pos = np.zeros(NUM_SWERVES)
        self.drive_vel = np.zeros(NUM_SWERVES)
        self.x = np.zeros(3)
        self.dx = np.zeros(3)

        self.base_target = np.zeros(3)

        # --- S-curve profiling state (kept; now optional per-command) ---
        self._smooth_active = False  # whether to apply smoothing for the *current* command
        self._v_prof = np.zeros(3, dtype=float)
        self._seg_v0 = np.zeros(3, dtype=float)
        self._seg_v1 = np.zeros(3, dtype=float)
        self._seg_t = 0.0
        self._seg_T = 0.0

        self._a_max = np.array([1.9, 1.9, 6.5], dtype=float)
        self._T_min = 0.01
        self._retarget_eps = 1e-3


        mujoco.mj_forward(self.model,self.data)

class YORGymEnv(gym.Env):
    env_limit = 10
    distance_threshold = 0.5
    def __init__(self,max_steps=30,
                use_orientation=False,noise_scale=0.01,
                return_full_trajectory=False, max_speed=1.0, max_steering_angle=1.0,prop_steps=100):
        self.max_steps = max_steps
        self.yor = BaseMujoco((_HERE / "yor-description" / "scene.mjcf").as_posix(),self.env_limit)
        self.yor.reset()
        self.action_space = spaces.Box(low=-1,high=1,shape=(3,))

        self.obs_dims = 3
        self.goal_dims = 3 if use_orientation else 2

        self.observation_space = spaces.Dict({
            "observation": spaces.Box(low=-np.inf,high=np.inf,shape=(self.obs_dims,)),
            "achieved_goal": spaces.Box(low=-np.inf,high=np.inf,shape=(self.goal_dims,)),
            "desired_goal": spaces.Box(low=-np.inf,high=np.inf,shape=(self.goal_dims,))
        })

        self.use_orientation = use_orientation
        self.return_full_trajectory = return_full_trajectory
        
        self.max_speed = max_speed
        self.max_steering_angle = max_steering_angle

        self.prop_steps = prop_steps

    def reset(self,goal=None):
        self.yor.reset()
        self.steps = 0
        if goal is  None:
            self.goal = np.random.uniform(-self.env_limit,self.env_limit,size=(self.goal_dims,))
            if self.use_orientation:
                self.goal[2] = np.random.uniform(-np.pi,np.pi)
        else:
            self.goal = goal
        return self._get_obs()

    def _get_obs(self):
        obs = self.yor.get_obs()
        
        if self.use_orientation:
            achieved_goal = np.array([obs[0],obs[1],quat2euler(obs[3:7])[2]])
        else:
            achieved_goal = np.array([obs[0],obs[1]])  
        return {
            "observation": np.float32(obs),
            "achieved_goal": np.float32(achieved_goal),
            "desired_goal": np.float32(self.goal)
        } 

    def _terminal(self,s,g):
        return goal_distance(s,g) < self.distance_threshold

    def compute_reward(self,ag,dg,info):
        return -(goal_distance(ag,dg) >= self.distance_threshold).astype(np.float32)

    def step(self,action):
        self.steps += 1
        
        applied_action = np.zeros_like(action)
        applied_action[0] = action[0]*self.max_steering_angle
        applied_action[1] = action[1]*self.max_speed
        self.yor.apply_action(action)
        
        current_traj = []
        for _ in range(self.prop_steps):
            for i in range(self.yor.model.nv): self.yor.data.qacc_warmstart[i] = 0 
            mujoco.mj_step(self.yor.model,self.yor.data)
            if self.return_full_trajectory:
                current_traj.append(self._get_obs()["achieved_goal"])
        obs = self._get_obs()
        info = {
            "is_success": self._terminal(obs["achieved_goal"],obs["desired_goal"]),
            "traj": np.array(current_traj)
        }
        done = self._terminal(obs["achieved_goal"],obs["desired_goal"]) or self.steps >= self.max_steps
        reward = self.compute_reward(obs["achieved_goal"],obs["desired_goal"],{})
        return obs,reward,done,info

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

def quat2euler(q_mj):
    q_scipy = np.array([q_mj[1],q_mj[2],q_mj[3],q_mj[0]])
    r = R.from_quat(q_scipy)
    return r.as_euler('xyz',degrees=False)


if __name__ == "__main__":     
    env = YORGymEnv()
    obs = env.reset()
    traj = [np.copy(obs["observation"])]
    for _ in range(300):
        action = env.action_space.sample()
        action = np.array([0.0,1.0])
        obs, reward, done, _ = env.step(action)
        traj.append(np.copy(obs["observation"]))
        print("Achieved: ",obs["achieved_goal"])
        print("Desired: ",obs["desired_goal"])
        print("Reward: ",reward)
        print("==========================================")
        if done: 
            print("Done")
            break
    
    traj = np.array(traj)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,8))
    plt.xlim(-env.env_limit,env.env_limit)
    plt.ylim(-env.env_limit,env.env_limit)
    traj = np.vstack(traj)
    plt.plot(traj[:,0],traj[:,1])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig("env_test.png")
    print(traj[:,0],traj[:,1])


