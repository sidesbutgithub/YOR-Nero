# joystick.py
"""
To run joystick on Jetson Nano, install xpad drivers, add user to the input group,
and set appropriate udev rules.
"""

import time
import numpy as np
import pygame
from pygame.joystick import Joystick
from loop_rate_limiters import RateLimiter

from commlink import RPCClient

XBOX_CONTROLLER_MAP = {
    "start": 7,
    "back": 6,
    "l1": 4,
    "r1": 5,
    "left_horizontal_axis": 0,
    "left_vertical_axis": 1,
    "right_horizontal_axis": 3,
    "right_vertical_axis": 4,
}

PS4_CONTROLLER_MAP = {
    "start": 3,
    "back": 0,  # square
    "l1": 4,
    "r1": 5,
    "left_horizontal_axis": 0,
    "right_horizontal_axis": 3,
    "right_vertical_axis": 4,
    "pad_y": 7,
}

controller_map = XBOX_CONTROLLER_MAP


def apply_deadzone(arr, dz=0.05):
    return np.where(np.abs(arr) <= dz, 0.0, np.sign(arr) * (np.abs(arr) - dz) / (1 - dz))


class JoystickNode:
    def __init__(self):
        pygame.init()
        if pygame.joystick.get_count() < 1:
            raise RuntimeError("No joystick detected")
        self.joystick = Joystick(0)

        # Three speed presets (m/s, m/s, rad/s) for Base.set_target_base_velocity
        self.max_vels = [
            np.array([0.50, 0.50, 1.57]),
            np.array([0.25, 0.25, 1.57]),
            np.array([0.75, 0.75, 1.57]),
        ]
        self.max_vel_setting = 0
        self.vel_alpha = 0.9  # low-pass on commanded vel

        self.control_loop_running = False

        # RPC to YOR (which wraps the new Base)
        self.yor = RPCClient(host="localhost", port=5557)
        self.yor.init()  # starts Base control loop on the server

        # D-pad debug state
        self.last_pad_y = 0
        self.last_pad_x = 0
        self.warned_no_hat = False

    def display_joystick_inputs(self):
        """Display current joystick input values"""
        print("\n" + "="*50)
        print("JOYSTICK INPUTS:")
        print("="*50)
        
        # Display axis values
        print("AXES:")
        for i in range(self.joystick.get_numaxes()):
            axis_value = self.joystick.get_axis(i)
            print(f"  Axis {i}: {axis_value:6.3f}")
        
        # Display button states
        print("BUTTONS:")
        for i in range(self.joystick.get_numbuttons()):
            button_state = self.joystick.get_button(i)
            print(f"  Button {i}: {'PRESSED' if button_state else 'RELEASED'}")
        
        # Display hat values
        num_hats = self.joystick.get_numhats()
        if num_hats > 0:
            print("HATS:")
        for i in range(num_hats):
                hat_value = self.joystick.get_hat(i)
                print(f"  Hat {i}: {hat_value}")
        else:
            print("HATS: None available")
        
        # Display current control state
        print(f"CONTROL STATE: {'RUNNING' if self.control_loop_running else 'STOPPED'}")
        print(f"MAX VEL SETTING: {self.max_vel_setting}")
        print("="*50)

    def control_loop(self):
        rate = RateLimiter(60, name="joystick")
        last_target_velocity = np.zeros(3, dtype=float)
        display_counter = 0  # Counter to control display frequency

        while True:
            pygame.event.pump()
            
            if not self.control_loop_running and self.joystick.get_button(controller_map["start"]):
                self.control_loop_running = True
                print("Control started")
                self.display_joystick_inputs()

            if self.control_loop_running and self.joystick.get_button(controller_map["back"]):
                self.control_loop_running = False
                print("Control stopped")
                self.display_joystick_inputs()

            if self.control_loop_running:
                left_bumper = self.joystick.get_button(controller_map["l1"])
                right_bumper = self.joystick.get_button(controller_map["r1"])
                
                if left_bumper:
                    self.max_vel_setting = (self.max_vel_setting + 1) % len(self.max_vels)
                    print("Max velocity setting:", self.max_vel_setting)
                    time.sleep(0.1)
                
                # Display joystick inputs when right bumper is pressed
                if right_bumper:
                    self.display_joystick_inputs()
                    time.sleep(0.1)
                
                # Display inputs every 5 seconds (300 frames at 60Hz)
                display_counter += 1
                if display_counter >= 300:
                    self.display_joystick_inputs()
                    display_counter = 0

                # Right stick → translation (vx, vy), Left stick X → yaw rate
                vy = -self.joystick.get_axis(controller_map["right_horizontal_axis"])
                vx = -self.joystick.get_axis(controller_map["right_vertical_axis"])
                w  = -self.joystick.get_axis(controller_map["left_horizontal_axis"])
                target_velocity = np.array([vx, vy, w], dtype=float)
                target_velocity = apply_deadzone(target_velocity)

                # scale & smooth
                target_velocity = self.max_vels[self.max_vel_setting] * target_velocity
                target_velocity = (1 - self.vel_alpha) * target_velocity + self.vel_alpha * last_target_velocity
                last_target_velocity = target_velocity

                # Send to Base (SparkFlex swerve)
                if np.linalg.norm(target_velocity, ord=1) > 1e-2:
                    self.yor.set_base_velocity(target_velocity)

                # D-pad up/down controls lift via Pico serial
                num_hats = self.joystick.get_numhats()
                if num_hats < 1:
                    if not self.warned_no_hat:
                        print("WARN: Controller reports 0 hats; D-pad unsupported by this device/driver")
                        self.warned_no_hat = True
                    pad_y = 0
                else:
                    pad = self.joystick.get_hat(0)
                    pad_x = pad[0]
                    pad_y = pad[1]

                if pad_y != self.last_pad_y:
                    print(f"D-pad Y changed: {self.last_pad_y} -> {pad_y}")
                    if pad_y > 0:
                        print("RPC: lift_up()")
                        self.yor.lift_up()
                    elif pad_y < 0:
                        print("RPC: lift_down()")
                        self.yor.lift_down()
                    else:
                        print("RPC: lift_stop()")
                        self.yor.lift_stop()
                        self.last_pad_y = pad_y

                if pad_x != self.last_pad_x:
                    print(f"D-pad Y changed: {self.last_pad_x} -> {pad_x}")
                    if pad_x > 0:
                        print("Arm: Home")
                        self.yor.left_arm.home()
                    elif pad_x < 0:
                        print("Arm: Home")
                        self.yor.left_arm.home()
                    else:
                        print("Arm: None")
                        
                    self.last_pad_x = pad_x


                
            rate.sleep()


def main():
    node = JoystickNode()
    node.control_loop()


if __name__ == "__main__":
    main()
