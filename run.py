import time
import argparse

from utils.wrappers import *
from utils.utils import *
from utils.param_dict import param_dict
from models.DDPG import DDPG
from models.DQN import DQNAgent


parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, required=True, choices=['VideoPinball-ramNoFrameskip-v4', 'BreakoutNoFrameskip-v4', 'PongNoFrameskip-v4', 'BoxingNoFrameskip-v4',
                                                                                   'Hopper-v2', 'Humanoid-v2', 'HalfCheetah-v2', 'Ant-v2'])
parser.add_argument('--to_train', action='store_true')
args = parser.parse_args()


if args.env_name in ['VideoPinball-ramNoFrameskip-v4', 'BreakoutNoFrameskip-v4', 'PongNoFrameskip-v4', 'BoxingNoFrameskip-v4']:
    # DQN
    lr = param_dict[args.env_name]['lr']
    num_episodes = param_dict[args.env_name]['num_episodes']
    gamma = param_dict[args.env_name]['gamma']
    eps_start = param_dict[args.env_name]['eps_start']
    eps_end = param_dict[args.env_name]['eps_end']
    eps_decay = 2 * (eps_start - eps_end) / num_episodes
    buffer_size = param_dict[args.env_name]['buffer_size']
    minimal_size = param_dict[args.env_name]['minimal_size']
    batch_size = param_dict[args.env_name]['batch_size']
    action_space = param_dict[args.env_name]['action_space']
    state_space = param_dict[args.env_name]['state_space']
    eval_eps = param_dict[args.env_name]['eval_eps']
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    env = gym.make(args.env_name)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    env.seed(0)
    if param_dict[args.env_name]['format'] == 'image':
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        env = FireResetEnv(env)
        env = WarpFrame(env)
        env = PyTorchFrame(env)
        env = ClipRewardEnv(env)
        env = FrameStack(env, 4)

    cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))

    if args.to_train:
        replay_buffer = ReplayBuffer(buffer_size)
        agent = DQNAgent(state_space, action_space, use_double_dqn=True, lr=lr, gamma=gamma, eps=eps_start,
                         format=param_dict[args.env_name]['format'])  # , pretrained='pretrained/checkpoint_dqn.pth'
        save_dir = f'logs/{args.env_name}/train_{cur_time}'
        os.makedirs(save_dir, exist_ok=True)
        log_file = os.path.join(save_dir, f'log.txt')
        return_list = train_off_policy_atari(env, agent, num_episodes, replay_buffer, minimal_size, batch_size, eps_decay, eps_end,
                                         file=log_file, save_dir=save_dir)
    else:
        agent = DQNAgent(state_space, action_space, use_double_dqn=True, lr=lr, gamma=gamma, eps=eps_start,
                                          format=param_dict[args.env_name]['format'], pretrained=f'checkpoints/{args.env_name}/DQN.pth')
        save_dir = f'logs/{args.env_name}/eval_{cur_time}'
        os.makedirs(save_dir, exist_ok=True)
        log_file = os.path.join(save_dir, f'log.txt')
        eval_atari(env, agent, 5, eval_eps, log_file)
else:
    # DDPG
    actor_lr = param_dict[args.env_name]['actor_lr']
    critic_lr = param_dict[args.env_name]['critic_lr']
    hidden_dim = param_dict[args.env_name]['hidden_dim']

    num_episodes = param_dict[args.env_name]['num_episodes']
    gamma = param_dict[args.env_name]['gamma']
    tau = param_dict[args.env_name]['tau'] # soft update
    buffer_size = param_dict[args.env_name]['buffer_size']
    minimal_size = param_dict[args.env_name]['minimal_size']
    batch_size = param_dict[args.env_name]['batch_size']
    sigma_start = param_dict[args.env_name]['sigma_start'] # Gaussian distribution standard variance
    sigma_end = param_dict[args.env_name]['sigma_end']
    sigma_decay = 2 * (sigma_start - sigma_end) / num_episodes
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = gym.make(args.env_name)
    random.seed(0)
    np.random.seed(0)
    env.seed(0)
    torch.manual_seed(0)

    state_space = param_dict[args.env_name]['state_space']
    action_space = param_dict[args.env_name]['action_space']
    action_bound = param_dict[args.env_name]['action_bound']

    cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    if args.to_train:
        replay_buffer = ReplayBuffer(buffer_size)
        agent = DDPG(state_space, action_space, hidden_dim, action_bound, gamma, tau, sigma_start, actor_lr, critic_lr, device)
        save_dir = f'logs/{args.env_name}/train_{cur_time}'
        os.makedirs(save_dir, exist_ok=True)
        log_file = os.path.join(save_dir, f'log.txt')
        return_list = train_off_policy_mujoco(env, state_space, agent, num_episodes, replay_buffer, minimal_size, sigma_decay, sigma_end, batch_size, file=log_file, save_dir=save_dir)
    else:
        agent = DDPG(state_space, action_space, hidden_dim, action_bound, gamma, tau, sigma_start, actor_lr, critic_lr, device, pretrained=f'checkpoints/{args.env_name}')
        save_dir = f'logs/{args.env_name}/eval_{cur_time}'
        os.makedirs(save_dir, exist_ok=True)
        log_file = os.path.join(save_dir, f'log.txt')
        eval_mujoco(env, state_space, agent, 5, log_file)