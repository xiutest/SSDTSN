import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams['font.family'] = 'Times New Roman'

file_paths = [
    'test_1_d3qn_rewards.txt',
    'test_1_dqn_rewards.txt',
    'test_2_d3qn_rewards.txt',
    'test_2_dqn_rewards.txt',
]

num_tests = 2
files_per_test = 2

grouped_file_paths = [file_paths[i * files_per_test:(i + 1) * files_per_test] for i in range(num_tests)]

grouped_labels = [
    ['G-D3QN Test 1', 'DQN Test 1'],
    ['G-D3QN Test 2', 'DQN Test 2'],
]

variant_colors = {
    'D3QN': '#384d70',
    'DQN': '#b32c3f',
}

def read_and_process(file_path, window_size=101, target_length=6000):
    """
    Read a text file and process the data, including removing outliers, applying moving average smoothing, and aligning data length.
    """
    try:
        with open(file_path, 'r') as file:
            data = file.read().splitlines()
            rewards = list(map(float, data))
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return np.zeros(target_length)
    except ValueError:
        print(f"Non-numeric data found in file: {file_path}")
        return np.zeros(target_length)

    filtered_rewards = [x + 0 for x in rewards]

    if len(filtered_rewards) >= window_size:
        rewards_smoothed = np.convolve(filtered_rewards, np.ones(window_size) / window_size, mode='valid')
    else:
        rewards_smoothed = np.array(filtered_rewards)

    current_length = len(rewards_smoothed)
    if current_length >= target_length:
        aligned_rewards = rewards_smoothed[:target_length]
    elif current_length > 0:
        x_old = np.linspace(0, 1, num=current_length)
        x_new = np.linspace(0, 1, num=target_length)
        aligned_rewards = np.interp(x_new, x_old, rewards_smoothed)
    else:
        aligned_rewards = np.zeros(target_length)

    return aligned_rewards

all_grouped_rewards = []
for test_idx, test_files in enumerate(grouped_file_paths):
    test_rewards = []
    for file_path in test_files:
        rewards = read_and_process(file_path)
        test_rewards.append(rewards)
    all_grouped_rewards.append(test_rewards)

target_length = 6000
x_new = range(1, target_length + 1)

output_dir = 'plots'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Directory created: {output_dir}")

fig, axes = plt.subplots(nrows=1, ncols=num_tests, figsize=(18, 6), sharey=True)

for test_idx, (test_rewards, test_labels) in enumerate(zip(all_grouped_rewards, grouped_labels)):
    ax = axes[test_idx]

    for rewards, label in zip(test_rewards, test_labels):
        if 'D3QN' in label:
            color = variant_colors['D3QN']
        elif 'DQN' in label:
            color = variant_colors['DQN']
        else:
            color = '#000000'

        ax.plot(x_new, rewards, label=label, color=color)

    if test_idx == 0:
        ax.set_ylabel('Reward', fontsize=16)
    ax.set_xlabel('Episode', fontsize=16)

    if any(rewards.size > 0 for rewards in test_rewards):
        max_reward = max(rewards.max() for rewards in test_rewards)
    else:
        max_reward = 1
    ax.set_ylim(top=max_reward + 1.5)

    ax.set_xlim(0, target_length + 100)

    ax.set_xticks(range(0, target_length + 1, 1000))
    ax.tick_params(axis='both', labelsize=12)

    ax.legend(loc='upper right', frameon=False, fontsize=14)

plt.tight_layout()
plt.show(block=False)

eps_path = os.path.join(output_dir, 'D3QN_vs_DQN_training_reward_comparison.eps')
plt.savefig(eps_path, format='eps', dpi=600, facecolor='white')
print(f"Image saved as '{eps_path}'.")

plt.close()
print("All images have been displayed and saved, the program continues to run...")
