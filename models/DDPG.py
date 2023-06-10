import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class QValueNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.linear1 = nn.Linear(state_dim+action_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        # self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        # self.linear4 = nn.Linear(hidden_dim, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, 1)

    def forward(self, s, a):
        # x1 = torch.cat([s, a], 1)
        # x2 = F.relu(self.linear1(x1))
        # x3 = F.relu(self.linear2(x2))
        # x4 = F.relu(self.linear3(x3 + x2))
        # x5 = F.relu(self.linear4(x4 + x3 + x2))
        # x6 = self.linear3(x5 + x4 + x3 + x2)

        x = torch.cat([s, a], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x6 = self.linear5(x)

        return x6


class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, action_bound=2):
        super(PolicyNet, self).__init__()
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        # self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        # self.linear4 = nn.Linear(hidden_dim, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound

    def forward(self, s):
        x1 = F.relu(self.linear1(s))
        x2 = F.relu(self.linear2(x1))
        # # x3 = F.relu(self.linear3(x2 + x1))
        # # x4 = F.relu(self.linear4(x3 + x2 + x1))
        # x5 = torch.tanh(self.linear5(x4 + x3 + x2 + x1)) * self.action_bound
        x5 = torch.tanh(self.linear5(x2)) * self.action_bound
        return x5


class DDPG:
    def __init__(self, state_dim, action_dim, hidden_dim, action_bound, gamma=0.95, tau=0.01, sigma_start=0.2, actor_lr=3.e-4,
                 critic_lr=3.e-3, device='cuda', pretrained=None,):
        self.actor = PolicyNet(state_dim, action_dim, hidden_dim, action_bound).to(device)
        self.actor_target = PolicyNet(state_dim, action_dim, hidden_dim, action_bound).to(device)
        self.critic = QValueNet(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = QValueNet(state_dim, action_dim, hidden_dim).to(device)

        if pretrained is not None:
            print(f"Loading a policy - {pretrained} ")
            print(self.actor.load_state_dict(torch.load(f'{pretrained}/DDPG_actor.pth')))
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.sigma = sigma_start
        self.action_dim = action_dim
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), critic_lr)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_target.load_state_dict(self.actor.state_dict())

    def take_action(self, state):
        action = self.actor(torch.tensor(state, dtype=torch.float, device=self.device).view(1, -1))
        action = action.cpu().detach().numpy()
        action += np.random.randn(self.action_dim) * self.sigma

        return action

    def soft_update(self, net, target_net):
        for parameter, target_parameter in zip(net.parameters(), target_net.parameters()):
            target_parameter.data.copy_(target_parameter.data * (1 - self.tau) + parameter.data * self.tau)

    def update(self, transition_dict):
        b = len(transition_dict['states'])
        state = torch.tensor(np.concatenate(transition_dict['states']), dtype=torch.float, device=self.device).reshape(b, -1)
        action = torch.tensor(np.concatenate(transition_dict['actions']), dtype=torch.float).view(b, -1).to(self.device)
        reward = torch.tensor(transition_dict['rewards'],
                              dtype=torch.float).view(-1, 1).to(self.device)
        next_state = torch.tensor(np.concatenate(transition_dict['next_states']), dtype=torch.float32,
                         device=self.device).reshape(b, -1)
        done = torch.tensor(transition_dict['dones'],
                            dtype=torch.float).view(-1, 1).to(self.device)

        q_target = reward + self.gamma * self.critic_target(next_state, self.actor_target(next_state)) * (1 - done)
        q_value = self.critic(state, action)
        critic_loss = torch.mean(nn.functional.mse_loss(q_value, q_target))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        actor_loss = -torch.mean(self.critic(state, self.actor(state)))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)