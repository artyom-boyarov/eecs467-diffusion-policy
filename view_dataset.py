import torch
import matplotlib.pyplot as plt
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

repo_id = "aboyarov/eval_red-block-ar-tag-full"

# 1) Load from the Hub (cached locally)
dataset = LeRobotDataset(repo_id)

# 2) Extract initial joint states across all episodes
initial_joint_states = []
episode_indices = [i for i in range(len(dataset))]

for episode in dataset:
    if episode['timestamp'] == 0:  # Assuming the first observation in each episode corresponds to the initial state
        joint_state = episode['observation.state']
        initial_joint_states.append(joint_state)
        print(joint_state)

# 3) Convert to numpy array for plotting
initial_joint_states = np.array(initial_joint_states)
episode_indices = np.array(episode_indices)

print(f"\nTotal episodes: {len(initial_joint_states)}")
print(f"Joint state shape: {initial_joint_states.shape}")

x = [[e for _ in range(initial_joint_states.shape[1])] for e in episode_indices]

# 4) Plot scatter graph of initial joint states
if initial_joint_states.ndim == 2 and initial_joint_states.shape[1] >= 2:
    # Use first two dimensions for scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x, initial_joint_states.flatten())
    plt.xlabel("Joint State Dimension 1")
    plt.ylabel("Joint State Dimension 2")
    plt.title("Initial Joint States Across All Episodes")
    plt.colorbar(label="Episode Index")
    plt.grid(True, alpha=0.3)
    plt.savefig("outputs/initial_joint_states_scatter.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("\nScatter plot saved to outputs/initial_joint_states_scatter.png")
else:
    print(f"\nWarning: Cannot create 2D scatter plot. Joint state shape is {initial_joint_states.shape}")