"""Terminal controls for the local YOR MuJoCo simulation.

Start ``python -m robot.yor_mujoco`` first, then run this module in another
terminal. Commands are deliberately explicit so it also works over SSH.
"""

import argparse
import shlex
import time

from commlink import RPCClient


HELP = """Commands:
  base FORWARD LEFT YAW       set persistent body-frame velocity (m/s, m/s, rad/s)
  drive FORWARD LEFT YAW SEC  drive for a fixed duration, then stop
  stop                        stop the base immediately
  lift HEIGHT                 set lift extension (0 to 0.416 m)
  left|right home             send one arm to its home pose
  left|right joints Q1 ... Q7 set seven arm joint targets in radians
  status                      print command acknowledgement and current state
  help                        show this help
  quit                        exit the console (the simulator keeps running)
"""


def print_acknowledgement(response: dict):
    print(f"accepted command #{response['sequence']}: {response}")


def run_console(port: int):
    robot = RPCClient(host="localhost", port=port)
    print(HELP)
    while True:
        try:
            tokens = shlex.split(input("yor-sim> "))
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not tokens:
            continue
        command, *args = tokens
        try:
            if command in {"quit", "exit"}:
                return
            if command == "help":
                print(HELP)
            elif command == "base" and len(args) == 3:
                print_acknowledgement(robot.command_base(*(float(value) for value in args)))
            elif command == "drive" and len(args) == 4:
                print_acknowledgement(robot.command_base(*(float(value) for value in args[:3])))
                time.sleep(float(args[3]))
                print_acknowledgement(robot.command_stop())
            elif command == "stop" and not args:
                print_acknowledgement(robot.command_stop())
            elif command == "lift" and len(args) == 1:
                print_acknowledgement(robot.command_lift(float(args[0])))
            elif command in {"left", "right"} and args[:1] == ["home"] and len(args) == 1:
                print_acknowledgement(robot.command_arm_home(command))
            elif command in {"left", "right"} and args[:1] == ["joints"] and len(args) == 8:
                print_acknowledgement(robot.command_arm_joints(command, [float(value) for value in args[1:]]))
            elif command == "status" and not args:
                print(robot.command_status())
            else:
                print("Invalid command. Type 'help' for the command list.")
        except Exception as exc:
            print(f"Command failed: {exc}")


def main():
    parser = argparse.ArgumentParser(description="Control a running YOR MuJoCo simulation.")
    parser.add_argument("--port", type=int, default=5557)
    run_console(parser.parse_args().port)


if __name__ == "__main__":
    main()
