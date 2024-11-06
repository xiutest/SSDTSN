import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Times New Roman'

file_paths = [
    'rewards_weights_0.2_0.7_0.1.txt',
    'rewards_weights_0.3_0.5_0.2.txt',
    'rewards_weights_0.4_0.4_0.2.txt',
    'rewards_weights_0.5_0.3_0.2.txt',
    'rewards_weights_0.7_0.2_0.1.txt'
]

labels = [
    '(μ, ν, λ) = (0.2, 0.7, 0.1)',
    '(μ, ν, λ) = (0.3, 0.5, 0.2)',
    '(μ, ν, λ) = (0.4, 0.4, 0.2)',
    '(μ, ν, λ) = (0.5, 0.3, 0.2)',
    '(μ, ν, λ) = (0.7, 0.2, 0.1)'
]

colors = ['#e6b733', '#384d70', '#b32c3f', '#9e7ea6', '#469161']

rewards_list = []

for file_path in file_paths:
    with open(file_path, 'r') as file:
        data = file.read().splitlines()
        rewards = list(map(float, data))
        filtered_rewards = [x + 0 for x in rewards]
        window_size = 100
        rewards_smoothed = np.convolve(filtered_rewards, np.ones(window_size)/window_size, mode='valid')
        rewards_list.append(rewards_smoothed)

target_length = 6000
aligned_rewards_list = [rewards[:target_length] for rewards in rewards_list]

plt.figure(figsize=(10, 6))

for i, (rewards, label, color) in enumerate(zip(aligned_rewards_list, labels, colors)):
    x = range(1, len(rewards) + 1)
    plt.plot(x, rewards, label=label, color=color)

plt.xlabel('Episode', fontsize=14)
plt.ylabel('G-D3QN Reward', fontsize=14)

plt.xticks(range(0, 6001, 1000), fontsize=12)
plt.yticks(fontsize=12)

max_reward = max([max(rewards) for rewards in aligned_rewards_list])
plt.ylim(top=max_reward + 3)

plt.xlim(right=6100)

plt.legend(loc='upper right', bbox_to_anchor=(0.98, 1), frameon=False, ncol=2, fontsize=12)

plt.show(block=False)
plt.pause(5)

plt.savefig('D3QN_reward_values_under_different_weights.png', bbox_inches='tight', dpi=600)
plt.savefig('D3QN_reward_values_under_different_weights.pdf', bbox_inches='tight', dpi=600)
plt.savefig('D3QN_reward_values_under_different_weights.svg', format='svg', bbox_inches='tight')
plt.savefig('D3QN_reward_values_under_different_weights.eps', format='eps', bbox_inches='tight')

plt.close()

print("The image has been displayed and saved, the program continues to run...")
