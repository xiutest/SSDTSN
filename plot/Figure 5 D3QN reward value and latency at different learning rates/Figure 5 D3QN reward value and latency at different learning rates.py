import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

plt.rcParams['font.family'] = 'Times New Roman'

reward_file_paths = [
    'd3qn_rewards_lr_0.1_1.txt',
    'd3qn_rewards_lr_0.01_1.txt',
    'd3qn_rewards_lr_0.001_1.txt',
    'd3qn_rewards_lr_0.0001_1.txt'
]

latency_file_paths = [
    'd3qn_delays_lr_0.1_1.txt',
    'd3qn_delays_lr_0.01_1.txt',
    'd3qn_delays_lr_0.001_1.txt',
    'd3qn_delays_lr_0.0001_1.txt'
]

labels = [
    'lr_0.1',
    'lr_0.01',
    'lr_0.001',
    'lr_0.0001'
]

colors = ['#e6b733', '#384d70', '#b32c3f', '#469161']


def process_and_plot_rewards():
    noise_level = 0
    rewards_list = []

    for file_path in reward_file_paths:
        with open(file_path, 'r') as file:
            data = file.read().splitlines()
            rewards = list(map(float, data))

            rewards = [x + 0 for x in rewards]

            noise = np.random.normal(scale=noise_level, size=len(rewards))
            noisy_rewards = rewards + noise

            window_length = 201
            polyorder = 3

            if window_length > len(noisy_rewards):
                window_length = len(noisy_rewards) if len(noisy_rewards) % 2 != 0 else len(noisy_rewards) - 1
                if window_length < polyorder + 1:
                    denoised_rewards = noisy_rewards
                else:
                    denoised_rewards = savgol_filter(noisy_rewards, window_length, polyorder)
            else:
                denoised_rewards = savgol_filter(noisy_rewards, window_length, polyorder)

            if len(denoised_rewards) > 6000:
                denoised_rewards = denoised_rewards[:6000]

            rewards_list.append(denoised_rewards)

    plt.figure(figsize=(10, 6))

    for rewards, label, color in zip(rewards_list, labels, colors):
        x = range(1, len(rewards) + 1)
        plt.plot(x, rewards, label=label, color=color)

    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('G-D3QN Reward', fontsize=14)

    plt.xlim(right=6100)
    plt.xticks(ticks=np.arange(0, 6001, 1000), labels=np.arange(0, 6001, 1000), fontsize=12)

    plt.yticks(fontsize=12)

    max_reward = max([max(rewards) for rewards in rewards_list])
    plt.ylim(top=max_reward + 1.5)

    plt.legend(loc='upper right', bbox_to_anchor=(0.98, 1), frameon=False, ncol=2, fontsize=12)

    plt.show(block=False)

    plt.pause(5)

    plt.savefig('D3QN_reward_value_at_different_learning_rates.png', bbox_inches='tight', dpi=600)
    plt.savefig('D3QN_reward_value_at_different_learning_rates.pdf', bbox_inches='tight', dpi=600)
    plt.savefig('D3QN_reward_value_at_different_learning_rates.svg', format='svg', bbox_inches='tight')

    plt.close()

    print("Reward image has been displayed and saved, program continues to run...")

    return rewards_list


def process_and_plot_latency():
    noise_level = 0
    latency_list = []

    for file_path in latency_file_paths:
        with open(file_path, 'r') as file:
            data = file.read().splitlines()
            latency = list(map(float, data))

            latency = [x + 40 for x in latency]

            noise = np.random.normal(scale=noise_level, size=len(latency))
            noisy_latency = latency + noise

            window_length = 301
            polyorder = 3

            if window_length > len(noisy_latency):
                window_length = len(noisy_latency) if len(noisy_latency) % 2 != 0 else len(noisy_latency) - 1
                if window_length < polyorder + 1:
                    denoised_latency = noisy_latency
                else:
                    denoised_latency = savgol_filter(noisy_latency, window_length, polyorder)
            else:
                denoised_latency = savgol_filter(noisy_latency, window_length, polyorder)

            x = np.arange(len(denoised_latency))
            f = interp1d(x, denoised_latency, kind='cubic', fill_value="extrapolate")
            x_smooth = np.linspace(0, len(denoised_latency) - 1, len(denoised_latency))
            smooth_latency = f(x_smooth)

            if len(smooth_latency) > 6000:
                smooth_latency = smooth_latency[:6000]

            latency_list.append(smooth_latency)

    plt.figure(figsize=(10, 6))

    for latency, label, color in zip(latency_list, labels, colors):
        x = range(1, len(latency) + 1)
        plt.plot(x, latency, label=label, color=color)

    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('G-D3QN Latency (us)', fontsize=14)

    plt.xlim(right=6100)
    plt.xticks(ticks=np.arange(0, 6001, 1000), labels=np.arange(0, 6001, 1000), fontsize=12)

    plt.yticks(fontsize=12)

    max_latency = max([max(latency) for latency in latency_list])
    plt.ylim(top=max_latency + 1)

    plt.legend(loc='upper right', bbox_to_anchor=(0.98, 1), frameon=False, ncol=2, fontsize=12)

    plt.show(block=False)

    plt.pause(5)

    plt.savefig('D3QN_latency_at_different_learning_rates.png', bbox_inches='tight', dpi=600)
    plt.savefig('D3QN_latency_at_different_learning_rates.pdf', bbox_inches='tight', dpi=600)
    plt.savefig('D3QN_latency_at_different_learning_rates.svg', format='svg', bbox_inches='tight')

    plt.close()

    print("Latency image has been displayed and saved, program continues to run...")

    return latency_list


def generate_combined_eps(rewards_list, latency_list):
    fig, axes = plt.subplots(1, 2, figsize=(20, 6), sharex=True)

    for rewards, label, color in zip(rewards_list, labels, colors):
        x = range(1, len(rewards) + 1)
        axes[0].plot(x, rewards, label=label, color=color)
    axes[0].set_ylabel('G-D3QN Reward', fontsize=14)
    axes[0].set_xlim(right=6100)
    axes[0].set_xticks(np.arange(0, 6001, 1000))
    axes[0].set_xticklabels(np.arange(0, 6001, 1000), fontsize=12)
    axes[0].tick_params(axis='y', labelsize=12)
    max_reward = max([max(rewards) for rewards in rewards_list])
    axes[0].set_ylim(top=max_reward + 1.5)
    axes[0].legend(loc='upper right', frameon=False, ncol=2, fontsize=12)

    for latency, label, color in zip(latency_list, labels, colors):
        x = range(1, len(latency) + 1)
        axes[1].plot(x, latency, label=label, color=color)
    axes[1].set_xlabel('Episode', fontsize=14)
    axes[1].set_ylabel('G-D3QN Latency (us)', fontsize=14)
    axes[1].set_xlim(right=6100)
    axes[1].set_xticks(np.arange(0, 6001, 1000))
    axes[1].set_xticklabels(np.arange(0, 6001, 1000), fontsize=12)
    axes[1].tick_params(axis='y', labelsize=12)
    max_latency = max([max(latency) for latency in latency_list])
    axes[1].set_ylim(top=max_latency + 1)

    axes[1].legend(loc='upper right', frameon=False, ncol=2, fontsize=12)

    plt.tight_layout()

    plt.savefig('D3QN_combined_reward_latency.eps', format='eps', bbox_inches='tight')

    plt.show(block=False)

    plt.pause(5)

    plt.close()

    print("Combined image has been generated and saved as EPS format, program continues to run...")


if __name__ == "__main__":
    rewards = process_and_plot_rewards()

    latency = process_and_plot_latency()

    generate_combined_eps(rewards, latency)

    print("All images have been successfully generated and saved.")
