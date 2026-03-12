import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.utils.constants import (
    OBS_ENV_STATE,
    OBS_STATE,
    ACTION,
    OBS_IMAGES,
    OBS_STR
)
import os
import pytorch_kinematics as pk
import shutil

DATASET_NAME = "eval_red-block-ar-tag-same-init"
NEW_DATASET_NAME = "eval_red-block-ar-tag-same-init-with-ee-pose"
HF_USER = os.getenv("HF_USER") or "aboyarov"
repo_id = HF_USER + "/" + DATASET_NAME

dataset = LeRobotDataset(repo_id, force_cache_sync=True)
new_dataset_features = dataset.features.copy()
EE_POSE_NAMES = ['x', 'y', 'z', 'r11', 'r12', 'r13', 'r21', 'r22', 'r23']
new_dataset_features["observation.ee_pose"] = {
    'dtype': 'float32', 
    'shape': (9,), # Stacked 3D position and continuous 6D rotation 
    'names': EE_POSE_NAMES
}
joint_names = new_dataset_features[ACTION]['names']

try:
    new_dataset = LeRobotDataset.create(
        repo_id=f"{HF_USER}/{NEW_DATASET_NAME}",
        fps=dataset.fps,
        features=new_dataset_features,
        robot_type=dataset.meta.robot_type,
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
            f"{HF_USER}/{NEW_DATASET_NAME}",
        )
    )
    new_dataset = LeRobotDataset.create(
        repo_id=f"{HF_USER}/{NEW_DATASET_NAME}",
        fps=dataset.fps,
        features=new_dataset_features,
        robot_type=dataset.meta.robot_type,
        use_videos=True,
        image_writer_threads=4,
    )

so101_chain = pk.build_serial_chain_from_urdf(
    open("so101_new_calib.urdf", encoding="utf-8").read(),
    end_link_name="gripper_link",
    root_link_name="base_link",
)

first_episode = True

for sample in dataset:
    if (sample['frame_index'] == 0).bool() and not first_episode:
        print("Saving episode", sample['episode_index'] - 1)
        new_dataset.save_episode()
    first_episode = False
    
    joint_obs = sample[OBS_STATE]
    ee_pose = so101_chain.forward_kinematics(joint_obs[:5])
    ee_pose_m = ee_pose.get_matrix()
    ee_rot_6d = ee_pose_m[0, :3, :2].flatten() # Take the first two columns of the rotation matrix and flatten to get 6D representation
    ee_pos = ee_pose_m[0, :3, 3].flatten() # Extract the translation part (
    ee_pose = torch.cat([ee_pos, ee_rot_6d]) # Concatenate position and rotation to get the final 9D pose representation

    action = {name: sample[ACTION][i] for i, name in enumerate(joint_names)}
    action_frame = build_dataset_frame(new_dataset_features, action, prefix=ACTION)

    obs_state = {name: sample[OBS_STATE][i] for i, name in enumerate(new_dataset_features[OBS_STATE]['names'])}
    ee_pose = {name: ee_pose[i] for i, name in enumerate(EE_POSE_NAMES)}
    observation_frame = build_dataset_frame(
        new_dataset_features, 
        {
            **obs_state,
            **ee_pose,
            "top": sample["observation.images.top"].view(480, 640, 3)
        }, 
        prefix=OBS_STR
    )

    frame = {**observation_frame, **action_frame, "task": "Red block on AR tag"}
    new_dataset.add_frame(frame)
new_dataset.save_episode()


new_dataset.finalize()
new_dataset.push_to_hub()