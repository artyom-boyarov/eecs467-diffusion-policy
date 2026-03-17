from lerobot.cameras.realsense import RealSenseCameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so_leader import SO101LeaderConfig, SO101Leader
from lerobot.processor.converters import (
    observation_to_transition,
    robot_action_observation_to_transition,
    transition_to_observation,
    transition_to_robot_action,
)
from lerobot.processor.core import RobotAction, RobotObservation
from lerobot.processor.pipeline import (
    IdentityProcessorStep,
    RobotProcessorPipeline,
    ProcessorStep,
    PipelineFeatureType,
    PolicyFeature,
    EnvTransition,
    ObservationProcessorStep,
)
from lerobot.policies.pretrained import PreTrainedPolicy, PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun
from lerobot.scripts.lerobot_record import record_loop
from lerobot.processor import make_default_processors
import os
import shutil
import time
import argparse

## In case you want to crop images
# IMG_TOP_LEFT = (0, 0)
# IMG_BOTTOM_RIGHT = (200, 500)

# class CropProcessor(ObservationProcessorStep):
#     def observation(self, observation: RobotObservation) -> RobotObservation:
#         observation['top'][
#             0:IMG_TOP_LEFT[0], :
#         ] = [0,0,0]
#         observation['top'][
#             IMG_BOTTOM_RIGHT[0]:-1, :
#         ] = [0,0,0]
#         observation['top'][
#             :, 0:IMG_TOP_LEFT[1]
#         ] = [0,0,0]
#         observation['top'][
#             :, IMG_BOTTOM_RIGHT[1]:-1
#         ] = [0,0,0]

#         return observation
#     def transform_features(
#         self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
#     ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
#         """Defines how this step modifies the description of pipeline features.

#         This method is used to track changes in data shapes, dtypes, or modalities
#         as data flows through the pipeline, without needing to process actual data.

#         Args:
#             features: A dictionary describing the input features for observations, actions, etc.

#         Returns:
#             A dictionary describing the output features after this step's transformation.
#         """
#         return features

NUM_EPISODES = 60
FPS = 30
EPISODE_TIME_SEC = 20
RESET_TIME_SEC = 5
TASK_DESCRIPTION = "Place red block on white square without occlusion"


def parse_args():
    """Parse command line arguments for recording configuration."""
    parser = argparse.ArgumentParser(
        description="Record LeRobot dataset episodes with teleoperation"
    )
    parser.add_argument(
        "--dataset-name",
        required=True,
        help="Name of the dataset (hub repo suffix) to create/overwrite",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=NUM_EPISODES,
        help=f"Number of episodes to record (default {NUM_EPISODES})",
    )
    parser.add_argument(
        "--episode-time",
        type=float,
        default=EPISODE_TIME_SEC,
        help=f"Length of each episode in seconds (default {EPISODE_TIME_SEC})",
    )
    parser.add_argument(
        "--task-description",
        type=str,
        default=TASK_DESCRIPTION,
        help="A short description of the task being demonstrated",
    )
    parser.add_argument(
        "--reset-time",
        type=float,
        default=RESET_TIME_SEC,
        help=f"Pause after resetting between episodes (default {RESET_TIME_SEC})",
    )
    return parser.parse_args()



def main():
    args = parse_args()

    # overwrite globals with user-specified values
    global NUM_EPISODES, EPISODE_TIME_SEC, RESET_TIME_SEC, TASK_DESCRIPTION
    NUM_EPISODES = args.num_episodes
    EPISODE_TIME_SEC = args.episode_time
    RESET_TIME_SEC = args.reset_time
    TASK_DESCRIPTION = args.task_description

    # dataset configuration (HF user is fixed for now)
    global DATASET_NAME
    DATASET_NAME = args.dataset_name

    # Order of connection:
    # 1: Robot follower 1
    # 2: Robot leader 1
    # 3: Robot follower 2
    # 4: Robot leader 2

    # Create robot configuration
    robot1_config = SO101FollowerConfig(
        id="follower_arm",
        cameras={
            "top": RealSenseCameraConfig(
                serial_number_or_name="152222070462", 
                width=640, 
                height=480, 
                fps=FPS,
                use_depth=False
            ),
            "wrist": RealSenseCameraConfig(
                serial_number_or_name="409122274501", 
                width=640,
                height=480,
                fps=FPS,
                use_depth=False
            ),
            # "wrist": OpenCVCameraConfig(
            #     index_or_path="/dev/video4", width=640, height=480, fps=FPS
            # ),
        },
        port="/dev/ttyACM0",
    )
    robot2_config = SO101FollowerConfig(
        id="follower_arm_2",
        port="/dev/ttyACM2",
    )

    teleop1_config = SO101LeaderConfig(
        id="leader_arm",
        port="/dev/ttyACM1",
    )

    teleop2_config = SO101LeaderConfig(
        id="leader_arm_2",
        port="/dev/ttyACM3",
    )

    # Initialize the robot and teleoperator
    robot1 = SO101Follower(robot1_config)
    robot2 = SO101Follower(robot2_config)
    teleop1 = SO101Leader(teleop1_config)
    teleop2 = SO101Leader(teleop2_config)

    # Configure the dataset features
    robot1_action_features = hw_to_dataset_features(robot1.action_features, "action.robot1")
    
    robot2_action_features = hw_to_dataset_features(robot2.action_features, "action.robot2")
    robot1_obs_features = hw_to_dataset_features(robot1.observation_features, "observation.robot1")
    robot2_obs_features = hw_to_dataset_features(robot2.observation_features, "observation.robot2")
    dataset_features = {**robot1_action_features, **robot2_action_features, **robot1_obs_features, **robot2_obs_features}

    print("Dataset features:", dataset_features)
    # Create the dataset
    HF_USER = os.getenv("HF_USER") or "aboyarov"
    try:
        dataset = LeRobotDataset.create(
            repo_id=f"{HF_USER}/{DATASET_NAME}",
            fps=FPS,
            features=dataset_features,
            robot_type=robot1.name,
            use_videos=True,
            image_writer_threads=4,
        )
    except FileExistsError:
        shutil.rmtree(
            os.path.join(
                os.path.expanduser("~"),
                ".cache",
                "huggingface",
                "lerobot",
                f"{HF_USER}/{DATASET_NAME}",
            )
        )
        dataset = LeRobotDataset.create(
            repo_id=f"{HF_USER}/{DATASET_NAME}",
            fps=FPS,
            features=dataset_features,
            robot_type=robot1.name,
            use_videos=True,
            image_writer_threads=4,
        )

    # Initialize the keyboard listener and rerun visualization
    _, events = init_keyboard_listener()

    # Connect the robot and teleoperator
    robot1.connect()
    robot2.connect()
    teleop1.connect()
    teleop2.connect()

    INIT_ANGLES = {
        "shoulder_pan.pos": 0.0,
        "shoulder_lift.pos": 0,
        "elbow_flex.pos": 30.0,
        "wrist_flex.pos": 55.0,
        "wrist_roll.pos": 0.0,
        "gripper.pos": 3.0,
    }  # Starting joint configuration.
    robot1.send_action(INIT_ANGLES)
    robot2.send_action(INIT_ANGLES)

    time.sleep(RESET_TIME_SEC)
    # Create the required processors
    teleop_action_processor, robot_action_processor, robot_observation_processor = (
        make_default_processors()
    )

    episode_idx = 0
    while episode_idx < NUM_EPISODES and not events["stop_recording"]:
        robot1.send_action(INIT_ANGLES)
        robot2.send_action(INIT_ANGLES)
        log_say(f"Recording episode {episode_idx + 1} of {NUM_EPISODES}")

        record_loop(
            robots=[robot1, robot2],
            events=events,
            fps=FPS,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            teleop=[teleop1, teleop2],
            dataset=dataset,
            control_time_s=EPISODE_TIME_SEC,
            single_task=TASK_DESCRIPTION,
            display_data=False,
            # policy=policy
        )

        # Reset the environment if not stopping or re-recording
        if not events["stop_recording"] and (
            episode_idx < NUM_EPISODES - 1 or events["rerecord_episode"]
        ):
            log_say("Reset the environment")

            robot1.send_action(INIT_ANGLES)
            robot2.send_action(INIT_ANGLES)

        if events["rerecord_episode"]:
            log_say("Re-recording episode")
            events["rerecord_episode"] = False
            events["exit_early"] = False
            dataset.clear_episode_buffer()
            continue

        dataset.save_episode()
        _ = input("Press Enter when ready for the next episode...")
        episode_idx += 1

    # Clean up
    log_say("Stop recording")
    robot1.disconnect()
    teleop1.disconnect()
    robot2.disconnect()
    teleop2.disconnect()
    dataset.finalize()
    dataset.push_to_hub()


if __name__ == "__main__":
    main()
