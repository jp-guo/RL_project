import os.path
import collections
import torch
from tqdm import tqdm
import numpy as np
import random


def train_off_policy_mujoco(env, state_dim, agent, num_episodes, replay_buffer, minimal_size, sigma_decay, sigma_end, batch_size, file, save_dir):
    return_list = []
    best = -10000
    pbar = tqdm(range(num_episodes))
    os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)
    for episode in pbar:
        episode_return = 0
        state = env.reset()[:state_dim]
        done = False
        agent.sigma = max(agent.sigma - sigma_decay, sigma_end)
        while not done:
            action = agent.take_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = next_state[:state_dim]
            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            episode_return += reward
            if replay_buffer.size() > minimal_size:
                b_states, b_actions, b_rewards, b_next_states, b_dones = replay_buffer.sample(batch_size)
                transition_dict = {'states': b_states, 'actions': b_actions,
                                   'next_states': b_next_states, 'rewards': b_rewards, 'dones': b_dones}
                agent.update(transition_dict)
        return_list.append(episode_return)
        pbar.set_description(f'Iteration {episode+1}: return={episode_return}')
        with open(file, 'a') as f:
            print(episode_return, file=f)
        if episode_return > best:
            torch.save(agent.actor.state_dict(), os.path.join(save_dir, 'checkpoints', f'DDPG_actor.pth'))
            torch.save(agent.critic.state_dict(), os.path.join(save_dir, 'checkpoints', f'DDPG_critic.pth'))
            best = episode_return
        torch.save(agent.actor.state_dict(), os.path.join(save_dir, 'checkpoints', f'DDPG_actor_{episode}.pth'))
        torch.save(agent.critic.state_dict(), os.path.join(save_dir, 'checkpoints', f'DDPG_critic_{episode}.pth'))
    return return_list


def eval_mujoco(env, state_dim, agent, num_episodes, file):
    return_list = []
    pbar = tqdm(range(num_episodes))
    with open(file, 'w') as f:
        for episode in pbar:
            episode_return = 0
            agent.sigma = 0.001
            state = env.reset()[:state_dim]
            done = False
            while not done:
                action = agent.take_action(state)
                next_state, reward, done, _ = env.step(action)
                next_state = next_state[:state_dim]
                state = next_state
                episode_return += reward

            return_list.append(episode_return)
            pbar.set_description(f'Iteration {episode+1}: return={episode_return}')
            print(f'Iteration {episode+1}: return={episode_return}', file=f)

        print(f'Average return: {np.array(return_list).mean()}')
        print(f'Std: {np.array(return_list).std()}')
        print(f'Average return: {np.array(return_list).mean()}', file=f)


def train_off_policy_atari(env, agent, num_episodes, replay_buffer, minimal_size, batch_size, eps_decay, eps_end, file, save_dir):
    return_list = []
    best = -10000
    pbar = tqdm(range(num_episodes))
    os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)
    for episode in pbar:
        episode_return = 0
        state = env.reset() / 255.
        done = False
        agent.eps = max(agent.eps - eps_decay, eps_end)
        while not done:
            action = agent.take_action(state)
            next_state, reward, done, terminated = env.step(action)
            next_state = next_state / 255.
            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            episode_return += reward
            if replay_buffer.size() > minimal_size:
                b_states, b_actions, b_rewards, b_next_states, b_dones = replay_buffer.sample(batch_size)
                transition_dict = {'states': b_states, 'actions': b_actions,
                                   'next_states': b_next_states, 'rewards': b_rewards, 'dones': b_dones}
                agent.update(transition_dict)
        return_list.append(episode_return)
        pbar.set_description(f'Iteration {episode+1}: return={episode_return}')
        with open(file, 'a') as f:
            print(episode_return, file=f)
        if episode_return > best:
            torch.save(agent.policy_network.state_dict(), os.path.join(save_dir, 'checkpoints', f'DQN.pth'))
            best = episode_return
        torch.save(agent.policy_network.state_dict(), os.path.join(save_dir, 'checkpoints', f'DQN_{episode}.pth'))
    return return_list


def eval_atari(env, agent, num_episodes, eval_eps, file):
    return_list = []
    pbar = tqdm(range(num_episodes))
    with open(file, 'w') as f:
        for episode in pbar:
            episode_return = 0
            state = env.reset() / 255.
            done = False
            agent.eps =  eval_eps
            while not done:
                action = agent.take_action(state)
                next_state, reward, done, _ = env.step(action)
                next_state = next_state / 255.
                state = next_state
                episode_return += reward
            return_list.append(episode_return)
            pbar.set_description(f'Iteration {episode+1}: return={episode_return}')
            print(f'Iteration {episode + 1}: return={episode_return}', file=f)
        print(f'Average return: {np.array(return_list).mean()}')
        print(f'Std: {np.array(return_list).std()}')
        print(f'Average return: {np.array(return_list).mean()}', file=f)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), np.array(action), reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This Memory class is modified based on the original code from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def add(self, state, action, reward, next_state, done):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, (state, action, reward, next_state, done))   # set the max p for new p

    def sample(self, n):
        b_idx, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, 1))
        b_memory = [0 for _ in range(n)]
        pri_seg = self.tree.total_p / n       # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p     # for later calculate ISweight
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            b_idx[i], b_memory[i] = idx, data
        return b_memory

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

    def size(self):
        return self.tree.size()

class SumTree(object):
    """
    This SumTree code is a modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py

    Story data with its priority in the tree.
    """
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        self.data = [0 for _ in range(capacity)]  # for all transitions

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def size(self):
        return len(self.data)

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        # print(p)
        # print(tree_idx)
        # print(self.tree[tree_idx])
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root