# EECS 467: Diffusion Policy Manipulation with the SO-101 arms

## Week 1: SO-101 arm setup.
1. Follow the instructions from here: https://huggingface.co/docs/lerobot/so101#joint-1 to assemble the arm together. Pay attention to the correct [motor gearing](https://huggingface.co/docs/lerobot/so101#step-by-step-assembly-instructions) for each joint.
2. [Set up the motor IDs](https://huggingface.co/docs/lerobot/so101#2-set-the-motors-ids-and-baudrates).
3. Plugging in the robot will expose it over device `/dev/ttyACM{i}` where i is the order of the robot you install, starting from 0. If it does not appear uninstall `brltty` which interferes with the USB driver:
  ```
sudo apt remove brltty
```
5. Finally, [calibrate](https://huggingface.co/docs/lerobot/so101#calibrate) the arms.

## Week 2: LeRobot installation
1. Follow the [installation tutorial](https://huggingface.co/docs/lerobot/installation) and install from pip:
```
pip install 'lerobot[feetech]'
```
2. Make sure to run the following whenever you open a new terminal for lerobot:
```
conda activate lerobot
```
3. Install the drivers for (Intel Realsense Cameras)[https://github.com/realsenseai/librealsense/blob/master/doc/distribution_linux.md#installing-the-packages]
4. Now follow instructions for teleoperation, dataset recording, etc. in the (imitation learning tutorial)[https://huggingface.co/docs/lerobot/il_robots]

## Week 3: Training a model on Great Lakes
1. Fork this repo, then clone it locally:
```
git clone <your-forked-repo-url>
```
2. (Log in)[https://huggingface.co/docs/lerobot/il_robots#record-a-dataset] to Hugging Face and store your credentials in a file called `hf_creds`. You can use `sample_hf_creds` and re-name it.
3. There are helper scripts added such as `record.py` which makes recording a dataset easier, and `rerun.sh` which runs inference with a trained policy.
4. Clone the repo on Great Lakes - make sure it is public.
5. Copy the `hf_creds` file to Great Lakes. **Don't push it to GitHub**.
```
scp -r ./hf_creds <uniqname>@greatlakes.arc-ts.umich.edu:/home/<uniqname>/eecs467-diffusion-policy/
```
6. In great lakes, install lerobot.
7. Install the eecs 467 diffusion policy:
```
cd lerobot_policy_diffusion_eecs467
pip3 install -e .
```
8. Launch the training job:
```
conda activate lerobot # Ensures correct Python & ffmpeg version
sbatch diff_train.batch
```
9. Run `sq` to see the job status, and see the log in the home directory.

## Week 4: Running inference
1. Locally, simply edit `rerun.sh` to fit your desired policy name and run it for inference. You don't need to collect the teleoperation robots for this.
