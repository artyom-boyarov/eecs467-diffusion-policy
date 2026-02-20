from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.cameras.realsense import RealSenseCameraConfig
FPS=30
# Create robot configuration
robot_config = SO101FollowerConfig(
    id="follower_arm",
    cameras={
        "top": RealSenseCameraConfig(
            serial_number_or_name="152222070462", width=640, height=480, fps=FPS
        )
    },
    port="/dev/ttyACM0",
)
# Initialize the robot and teleoperator
robot = SO101Follower(robot_config)
# Connect the robot and teleoperator
robot.connect()

INIT_ANGLES = {
    "shoulder_pan.pos": 0.0,
    "shoulder_lift.pos": 0,
    "elbow_flex.pos": 30.0,
    "wrist_flex.pos": 55.0,
    "wrist_roll.pos": 0.0,
    "gripper.pos": 3.0,
}  # Starting joint configuration.
robot.send_action(INIT_ANGLES)