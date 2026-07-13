import mujoco
import mujoco.viewer
import mink
import time
import numpy as np
import threading
from typing import Optional
from pathlib import Path
import atexit

from loop_rate_limiters import RateLimiter

from robot.arm.ik_solver import SingleArmIK
from commlink import RPCServer


class YORMujoco:
    def __init__(self, mjcf_path: str, solver_dt: float = 0.01):
        self.mjcf_path = mjcf_path
        self.solver_dt = solver_dt

        # launch mujoco
        self.model = mujoco.MjModel.from_xml_path(self.mjcf_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco.viewer.launch_passive(
            model=self.model,
            data=self.data,
        )
        self.viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

        # initialize arm
        self.left_q_desired: Optional[np.ndarray] = None
        self.left_q_desired_lock = threading.Lock()
        self.left_ik_solver = SingleArmIK(
            (_HERE / "yor-description/nero-welded-base-and-lift.mjcf").as_posix(),
            solver_dt=self.solver_dt,
            joint_names=[
                "left_arm_joint1",
                "left_arm_joint2",
                "left_arm_joint3",
                "left_arm_joint4",
                "left_arm_joint5",
                "left_arm_joint6",
                "left_arm_joint7",
            ],
            ee_frame="left_arm_ee",
        )

        self.right_q_desired: Optional[np.ndarray] = None
        self.right_q_desired_lock = threading.Lock()
        self.right_ik_solver = SingleArmIK(
            (_HERE / "yor-description/nero-welded-base-and-lift.mjcf").as_posix(),
            solver_dt=self.solver_dt,
            joint_names=[
                "right_arm_joint1",
                "right_arm_joint2",
                "right_arm_joint3",
                "right_arm_joint4",
                "right_arm_joint5",
                "right_arm_joint6",
                "right_arm_joint7",
            ],
            ee_frame="right_arm_ee",
        )

        self.control_loop_thread: threading.Thread | None = threading.Thread(target=self.control_loop, daemon=True)
        self.control_loop_running = False

    def set_left_ee_target(self, ee_target: mink.SE3, gripper_target: float = 0.0, preview_time: float = 0.0):
        self.left_ik_solver.update_configuration(self.data.qpos.copy())
        qd, is_solved = self.left_ik_solver.solve_ik(ee_target)
        print(f"desired q: {np.round(qd, 4)} | is_solved: {is_solved}")
        with self.left_q_desired_lock:
            self.left_q_desired = qd

    def set_left_joint_target(self, joint_target: np.ndarray):
        with self.left_q_desired_lock:
            self.left_q_desired = joint_target

    def get_left_joint_positions(self) -> np.ndarray:
        return self.data.qpos.copy()[self.left_ik_solver.dof_ids]

    def get_left_ee_pose(self) -> mink.SE3:
        q = self.data.qpos.copy()
        self.left_ik_solver.update_configuration(q)
        return self.left_ik_solver.forward_kinematics()

    def set_right_ee_target(self, ee_target: mink.SE3, gripper_target: float = 0.0, preview_time: float = 0.0):
        self.right_ik_solver.update_configuration(self.data.qpos.copy())
        qd, is_solved = self.right_ik_solver.solve_ik(ee_target)
        print(f"desired q: {np.round(qd, 4)} | is_solved: {is_solved}")
        with self.right_q_desired_lock:
            self.right_q_desired = qd

    def set_right_joint_target(self, joint_target: np.ndarray):
        with self.right_q_desired_lock:
            self.right_q_desired = joint_target

    def get_right_joint_positions(self) -> np.ndarray:
        return self.data.qpos.copy()[self.right_ik_solver.dof_ids]

    def get_right_ee_pose(self) -> mink.SE3:
        q = self.data.qpos.copy()
        self.right_ik_solver.update_configuration(q)
        return self.right_ik_solver.forward_kinematics()

    def start_control(self):
        if self.control_loop_thread is None:
            print("To initiate a new control loop, create a new instance of ArmMujoco first")
            return
        # self.init()
        self.control_loop_running = True
        self.control_loop_thread.start()

    def stop_control(self):
        if self.control_loop_thread is None:
            print("Control thread not running")
            return
        self.control_loop_running = False
        self.control_loop_thread.join()
        self.control_loop_thread = None
        self.viewer.close()

    def init(self):
        self.start_control()
        time.sleep(0.1)
        # home
        q = self.data.qpos.copy()
        self.left_ik_solver.init(q)
        self.right_ik_solver.init(q)
        with self.left_q_desired_lock:
            self.left_q_desired = q[self.left_ik_solver.dof_ids]
        with self.right_q_desired_lock:
            self.right_q_desired = q[self.right_ik_solver.dof_ids]

    def home_left_arm(self):
        with self.left_q_desired_lock:
            self.left_q_desired = self.left_ik_solver.get_home_q()

    def home_right_arm(self):
        with self.right_q_desired_lock:
            self.right_q_desired = self.right_ik_solver.get_home_q()

    def control_loop(self):
        rate_limiter = RateLimiter(200)
        while self.control_loop_running:
            if self.left_q_desired is not None:
                with self.left_q_desired_lock:
                    self.data.ctrl[self.left_ik_solver.actuator_ids] = self.left_q_desired
            if self.right_q_desired is not None:
                with self.right_q_desired_lock:
                    self.data.ctrl[self.right_ik_solver.actuator_ids] = self.right_q_desired
            # step mujoco
            mujoco.mj_step(self.model, self.data)
            self.viewer.sync()
            rate_limiter.sleep()


if __name__ == "__main__":
    _HERE = Path(__file__).parent
    yor_mujoco = YORMujoco(mjcf_path=(_HERE / "yor-description" / "scene.mjcf").as_posix())
    rpc_server = RPCServer(yor_mujoco, 8081, threaded=False)
    atexit.register(rpc_server.stop)
    rpc_server.start()
