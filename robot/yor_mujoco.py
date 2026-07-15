"""Interactive MuJoCo simulation server for the YOR robot.

Run this module to open the MuJoCo viewer and expose a local RPC interface.
``robot.sim_console`` is a hardware-free client for the interface.
"""

import argparse
import atexit
import threading
import time
from pathlib import Path
from typing import Optional

import mink
import mujoco
import mujoco.viewer
import numpy as np
from commlink import RPCServer
from robot.arm.ik_solver import SingleArmIK


_HERE = Path(__file__).parent
_WHEEL_RADIUS = 0.0381
_HALF_LENGTH = 0.1225
_HALF_WIDTH = 0.170


class YORMujoco:
    """A full YOR visual simulator with base, lift, and bimanual arm commands.

    The base pose is integrated kinematically from the requested body-frame
    velocity. This makes desktop teleoperation deterministic; the wheel steer
    and drive actuators are still commanded so the swerve modules animate.
    Arms and lift use the actuators defined in ``robot.mjcf``.
    """

    def __init__(self, mjcf_path: str, solver_dt: float = 0.01, control_hz: float = 60.0):
        self.mjcf_path = mjcf_path
        self.solver_dt = solver_dt
        if control_hz <= 0:
            raise ValueError("control_hz must be positive")
        self.control_hz = float(control_hz)
        self.model = mujoco.MjModel.from_xml_path(self.mjcf_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco.viewer.launch_passive(model=self.model, data=self.data)
        self.viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

        self.left_q_desired: Optional[np.ndarray] = None
        self.right_q_desired: Optional[np.ndarray] = None
        self.left_q_desired_lock = threading.Lock()
        self.right_q_desired_lock = threading.Lock()
        ik_model = (_HERE / "yor-description" / "nero-welded-base-and-lift.mjcf").as_posix()
        self.left_ik_solver = SingleArmIK(
            ik_model,
            solver_dt=self.solver_dt,
            joint_names=[f"left_arm_joint{i}" for i in range(1, 8)],
            ee_frame="left_arm_ee",
        )
        self.right_ik_solver = SingleArmIK(
            ik_model,
            solver_dt=self.solver_dt,
            joint_names=[f"right_arm_joint{i}" for i in range(1, 8)],
            ee_frame="right_arm_ee",
        )
        self._left_joint_names = [f"left_arm_joint{i}" for i in range(1, 8)]
        self._right_joint_names = [f"right_arm_joint{i}" for i in range(1, 8)]
        self._left_qpos_adrs = self._joint_qpos_addresses(self._left_joint_names)
        self._right_qpos_adrs = self._joint_qpos_addresses(self._right_joint_names)
        self._left_arm_actuators = np.array(
            [self._actuator_id(f"{name}_pos") for name in self._left_joint_names], dtype=int
        )
        self._right_arm_actuators = np.array(
            [self._actuator_id(f"{name}_pos") for name in self._right_joint_names], dtype=int
        )

        base_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base")
        self._base_qpos_adr = int(self.model.jnt_qposadr[base_joint_id])
        self._base_dof_adr = int(self.model.jnt_dofadr[base_joint_id])
        self._base_initial_qpos = self.data.qpos[self._base_qpos_adr : self._base_qpos_adr + 7].copy()
        self._base_position = self._base_initial_qpos[:3].copy()
        self._base_yaw = 0.0
        self._base_velocity = np.zeros(3, dtype=float)
        self._base_velocity_lock = threading.Lock()
        self._lift_target = 0.0
        self._lift_lock = threading.Lock()
        self._command_sequence = 0
        self._last_command = "reset"
        self._command_lock = threading.Lock()

        self._steer_actuators = [
            self._actuator_id(name)
            for name in (
                "front_left_steer_ctrl",
                "back_left_steer_ctrl",
                "front_right_steer_ctrl",
                "back_right_steer_ctrl",
            )
        ]
        self._drive_actuators = [
            self._actuator_id(name)
            for name in (
                "drive_front_left_ctrl",
                "drive_back_left_ctrl",
                "drive_front_right_ctrl",
                "drive_back_right_ctrl",
            )
        ]
        self._lift_actuator = self._actuator_id("Lift")
        self.control_loop_thread: threading.Thread | None = None
        self.control_loop_running = False
        self._initialized = False

    def _actuator_id(self, name: str) -> int:
        actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if actuator_id < 0:
            raise ValueError(f"MuJoCo actuator not found: {name}")
        return actuator_id

    def _joint_qpos_addresses(self, names: list[str]) -> np.ndarray:
        return np.array(
            [self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in names],
            dtype=int,
        )

    @staticmethod
    def _solver_qpos_addresses(solver: SingleArmIK, names: list[str]) -> np.ndarray:
        return np.array(
            [solver.model.jnt_qposadr[mujoco.mj_name2id(solver.model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in names],
            dtype=int,
        )

    def _sync_ik_from_scene(self, solver: SingleArmIK, names: list[str], scene_qpos_adrs: np.ndarray) -> np.ndarray:
        """Copy named arm joints into the smaller fixed-base IK model."""
        q = solver.configuration.q.copy()
        q[self._solver_qpos_addresses(solver, names)] = self.data.qpos[scene_qpos_adrs]
        solver.configuration.update(q)
        return q

    @staticmethod
    def _solver_joint_values(solver: SingleArmIK, names: list[str], q: np.ndarray) -> np.ndarray:
        return q[YORMujoco._solver_qpos_addresses(solver, names)]

    def init(self):
        """Initialize all controls and start simulation stepping (idempotent)."""
        if self._initialized:
            return
        left_q = self._sync_ik_from_scene(self.left_ik_solver, self._left_joint_names, self._left_qpos_adrs)
        right_q = self._sync_ik_from_scene(self.right_ik_solver, self._right_joint_names, self._right_qpos_adrs)
        self.left_ik_solver.init(left_q)
        self.right_ik_solver.init(right_q)
        with self.left_q_desired_lock:
            self.left_q_desired = self._solver_joint_values(self.left_ik_solver, self._left_joint_names, left_q)
        with self.right_q_desired_lock:
            self.right_q_desired = self._solver_joint_values(self.right_ik_solver, self._right_joint_names, right_q)
        self._initialized = True
        self.start_control()

    def start_control(self):
        if self.control_loop_running:
            return
        self.control_loop_running = True
        self.control_loop_thread = threading.Thread(target=self.control_loop, daemon=True)
        self.control_loop_thread.start()

    def stop_control(self):
        self.control_loop_running = False
        if self.control_loop_thread is not None:
            self.control_loop_thread.join(timeout=2.0)
            self.control_loop_thread = None
        self.viewer.close()

    # ---- Base and lift -------------------------------------------------
    def set_base_velocity(self, velocity: np.ndarray):
        """Set body-frame ``[forward, left, yaw]`` velocity in m/s, m/s, rad/s."""
        velocity = np.asarray(velocity, dtype=float)
        if velocity.shape != (3,):
            raise ValueError("base velocity must have shape (3,)")
        with self._base_velocity_lock:
            self._base_velocity = velocity.copy()

    def _record_command(self, name: str) -> int:
        with self._command_lock:
            self._command_sequence += 1
            self._last_command = name
            return self._command_sequence

    def stop_base(self):
        self.set_base_velocity(np.zeros(3, dtype=float))

    def get_base_pose(self) -> np.ndarray:
        """Return simulated base ``[x, y, z, yaw]`` in world coordinates."""
        return np.array([*self._base_position, self._base_yaw], dtype=float)

    def set_lift_height(self, height: float):
        """Set total lift extension in metres (the model range is 0–0.416 m)."""
        with self._lift_lock:
            self._lift_target = float(np.clip(height, 0.0, 0.416))

    # ---- Primitive RPC command protocol --------------------------------
    # These methods deliberately accept and return only Python primitives.
    # That keeps UI, remote control, and a future Gym wrapper independent of
    # commlink's support for NumPy or Mink objects.
    def command_base(self, forward: float, left: float, yaw: float) -> dict:
        self.set_base_velocity(np.array([forward, left, yaw], dtype=float))
        sequence = self._record_command("base")
        return {"accepted": True, "sequence": sequence, "base_velocity": [float(forward), float(left), float(yaw)]}

    def command_stop(self) -> dict:
        self.stop_base()
        sequence = self._record_command("stop")
        return {"accepted": True, "sequence": sequence}

    def command_lift(self, height: float) -> dict:
        self.set_lift_height(height)
        sequence = self._record_command("lift")
        return {"accepted": True, "sequence": sequence, "lift_height": self._lift_target}

    def command_arm_joints(self, side: str, joints: list[float]) -> dict:
        if side not in {"left", "right"}:
            raise ValueError("side must be 'left' or 'right'")
        target = np.asarray(joints, dtype=float)
        if side == "left":
            self.set_left_joint_target(target)
        else:
            self.set_right_joint_target(target)
        sequence = self._record_command(f"{side}_arm_joints")
        return {"accepted": True, "sequence": sequence, "side": side, "joints": target.tolist()}

    def command_arm_home(self, side: str) -> dict:
        if side == "left":
            self.home_left_arm()
        elif side == "right":
            self.home_right_arm()
        else:
            raise ValueError("side must be 'left' or 'right'")
        sequence = self._record_command(f"{side}_arm_home")
        return {"accepted": True, "sequence": sequence, "side": side}

    def command_status(self) -> dict:
        with self._base_velocity_lock:
            velocity = self._base_velocity.tolist()
        with self._lift_lock:
            lift_height = self._lift_target
        with self._command_lock:
            sequence, last_command = self._command_sequence, self._last_command
        return {
            "sequence": sequence,
            "last_command": last_command,
            "base_pose": self.get_base_pose().tolist(),
            "base_velocity": velocity,
            "lift_height": lift_height,
            "left_joints": self.get_left_joint_positions().tolist(),
            "right_joints": self.get_right_joint_positions().tolist(),
            "control_loop_running": self.control_loop_running,
            "control_hz": self.control_hz,
        }

    def lift_up(self):
        self.set_lift_height(self._lift_target + 0.01)

    def lift_down(self):
        self.set_lift_height(self._lift_target - 0.01)

    def lift_stop(self):
        pass

    def _apply_base_command(self, dt: float):
        with self._base_velocity_lock:
            forward, left, yaw_rate = self._base_velocity.copy()

        cos_yaw, sin_yaw = np.cos(self._base_yaw), np.sin(self._base_yaw)
        self._base_position[0] += (cos_yaw * forward - sin_yaw * left) * dt
        self._base_position[1] += (sin_yaw * forward + cos_yaw * left) * dt
        self._base_yaw += yaw_rate * dt

        yaw_quat = np.array([np.cos(self._base_yaw / 2), 0.0, 0.0, np.sin(self._base_yaw / 2)])
        quat = np.empty(4)
        mujoco.mju_mulQuat(quat, yaw_quat, self._base_initial_qpos[3:7])
        qpos = self.data.qpos
        qpos[self._base_qpos_adr : self._base_qpos_adr + 3] = self._base_position
        qpos[self._base_qpos_adr + 3 : self._base_qpos_adr + 7] = quat
        self.data.qvel[self._base_dof_adr : self._base_dof_adr + 6] = 0.0

        # Animate and orient the four swerve modules.
        wheel_positions = (( _HALF_LENGTH, _HALF_WIDTH), (-_HALF_LENGTH, _HALF_WIDTH),
                           ( _HALF_LENGTH, -_HALF_WIDTH), (-_HALF_LENGTH, -_HALF_WIDTH))
        for steer_id, drive_id, (x, y) in zip(self._steer_actuators, self._drive_actuators, wheel_positions):
            wheel_vx = forward - yaw_rate * y
            wheel_vy = left + yaw_rate * x
            self.data.ctrl[steer_id] = np.arctan2(wheel_vy, wheel_vx)
            self.data.ctrl[drive_id] = np.hypot(wheel_vx, wheel_vy) / _WHEEL_RADIUS

    # ---- Arm API -------------------------------------------------------
    def set_left_ee_target(self, ee_target: mink.SE3, gripper_target: float = 0.0, preview_time: float = 0.0):
        self._sync_ik_from_scene(self.left_ik_solver, self._left_joint_names, self._left_qpos_adrs)
        qd, _ = self.left_ik_solver.solve_ik(ee_target)
        with self.left_q_desired_lock:
            self.left_q_desired = qd

    def set_right_ee_target(self, ee_target: mink.SE3, gripper_target: float = 0.0, preview_time: float = 0.0):
        self._sync_ik_from_scene(self.right_ik_solver, self._right_joint_names, self._right_qpos_adrs)
        qd, _ = self.right_ik_solver.solve_ik(ee_target)
        with self.right_q_desired_lock:
            self.right_q_desired = qd

    def set_left_joint_target(self, joint_target: np.ndarray):
        joint_target = np.asarray(joint_target, dtype=float)
        if joint_target.shape != (7,):
            raise ValueError("left joint target must have seven values")
        with self.left_q_desired_lock:
            self.left_q_desired = joint_target.copy()

    def set_right_joint_target(self, joint_target: np.ndarray):
        joint_target = np.asarray(joint_target, dtype=float)
        if joint_target.shape != (7,):
            raise ValueError("right joint target must have seven values")
        with self.right_q_desired_lock:
            self.right_q_desired = joint_target.copy()

    def get_left_joint_positions(self) -> np.ndarray:
        return self.data.qpos.copy()[self._left_qpos_adrs]

    def get_right_joint_positions(self) -> np.ndarray:
        return self.data.qpos.copy()[self._right_qpos_adrs]

    def get_left_ee_pose(self) -> mink.SE3:
        self._sync_ik_from_scene(self.left_ik_solver, self._left_joint_names, self._left_qpos_adrs)
        return self.left_ik_solver.forward_kinematics()

    def get_right_ee_pose(self) -> mink.SE3:
        self._sync_ik_from_scene(self.right_ik_solver, self._right_joint_names, self._right_qpos_adrs)
        return self.right_ik_solver.forward_kinematics()

    def home_left_arm(self):
        self.set_left_joint_target(np.zeros(7, dtype=float))

    def home_right_arm(self):
        self.set_right_joint_target(np.zeros(7, dtype=float))

    def control_loop(self):
        """Step and render at a modest wall-clock rate without busy waiting."""
        period = 1.0 / self.control_hz
        previous_step = time.monotonic()
        while self.control_loop_running and self.viewer.is_running():
            loop_start = time.monotonic()
            # Kinematic command integration follows wall time so control
            # remains responsive even when rendering at 60 Hz instead of the
            # MuJoCo model's much smaller physics timestep.
            dt = min(loop_start - previous_step, 0.1)
            previous_step = loop_start
            self._apply_base_command(dt)
            with self.left_q_desired_lock:
                if self.left_q_desired is not None:
                    self.data.ctrl[self._left_arm_actuators] = self.left_q_desired
            with self.right_q_desired_lock:
                if self.right_q_desired is not None:
                    self.data.ctrl[self._right_arm_actuators] = self.right_q_desired
            with self._lift_lock:
                self.data.ctrl[self._lift_actuator] = self._lift_target
            mujoco.mj_step(self.model, self.data)
            self.viewer.sync()
            remaining = period - (time.monotonic() - loop_start)
            if remaining > 0:
                time.sleep(remaining)
        self.control_loop_running = False


def main():
    parser = argparse.ArgumentParser(description="Launch the YOR MuJoCo simulation server.")
    parser.add_argument("--port", type=int, default=5557, help="local RPC port (default: 5557)")
    parser.add_argument("--control-hz", type=float, default=60.0, help="simulation and viewer update rate (default: 60)")
    args = parser.parse_args()
    simulator = YORMujoco((_HERE / "yor-description" / "scene.mjcf").as_posix(), control_hz=args.control_hz)
    simulator.init()
    server = RPCServer(simulator, args.port, threaded=False)
    atexit.register(simulator.stop_control)
    atexit.register(server.stop)
    print(f"YOR simulation ready on RPC port {args.port}. Run: python -m robot.sim_console")
    server.start()


if __name__ == "__main__":
    main()
