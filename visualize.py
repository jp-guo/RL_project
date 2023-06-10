import os

import matplotlib.pyplot as plt
import numpy as np


def moving_average(a, window_size=51):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

os.makedirs('figures', exist_ok=True)
# 'VideoPinball-ramNoFrameskip-v4', 'BreakoutNoFrameskip-v4', 'PongNoFrameskip-v4', 'BoxingNoFrameskip-v4', 'Ant-v2', 'HalfCheetah-v2', 'Hopper-v2', 'Humanoid-v2'
for env_name in ['Ant-v2']:
    returns = np.loadtxt(f'logs/{env_name}/logs.txt')
    # returns = np.loadtxt(f'logs/{env_name}/real.txt')
    # returns[0] = -2000
    plt.figure()
    plt.plot(range(len(returns)), returns, color='lightblue')
    plt.plot(range(len(returns)), moving_average(returns))
    plt.xlabel('Iteration')
    plt.ylabel('Returns')
    plt.title(env_name)
    plt.grid()
    # plt.savefig(f'figures/{env_name}.png', bbox_inches='tight')
    plt.show()
    plt.close()
