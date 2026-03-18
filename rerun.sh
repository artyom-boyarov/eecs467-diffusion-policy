rm -rf /home/artyom/.cache/huggingface/lerobot/aboyarov/eval_red-block-ar-tag-test
python3 move_to_init.py

## For 1 robot block pick task:
# lerobot-record --robot.type=so101_follower \
#  --robot.port=/dev/ttyACM0 --robot.id=follower_arm \
#  --robot.cameras="{ top: { type: intelrealsense, serial_number_or_name: 152222070462, width: 640, height: 480, fps: 30}, wrist: {type: intelrealsense, serial_number_or_name: 409122274501, width: 640, height: 480, fps: 30}}"  \
#  --display_data=false --dataset.repo_id=aboyarov/eval_red-block-ar-tag-test \
#  --dataset.single_task="Place the red block on the AR tag" \
#  --policy.path=aboyarov/redblock-2-cameras-ar-tag-diffusion-eecs467-v0 \
#  --dataset.episode_time_s=360

## For 2 robot bimanual handoff task:
lerobot-record --robot.type=so101_follower \
 --robot.port=/dev/ttyACM0 --robot.id=follower_arm \
 --robot.cameras="{ top: { type: intelrealsense, serial_number_or_name: 152222070462, width: 640, height: 480, fps: 30}, wrist: {type: intelrealsense, serial_number_or_name: 409122274501, width: 640, height: 480, fps: 30}}"  \
 --display_data=false --dataset.repo_id=aboyarov/eval_red-block-ar-tag-test \
 --dataset.single_task="Place the red block on the AR tag" \
 --policy.path=aboyarov/redblock-2-cameras-ar-tag-diffusion-eecs467-v0 \
 --dataset.episode_time_s=360