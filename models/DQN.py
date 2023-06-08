import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DQN(nn.Module):
    def __init__(self, observation_space, action_space, format='image'):
        super().__init__()
        self.format = format
        if format == 'image':
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels=observation_space, out_channels=16, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
                nn.ReLU()
            )

            self.fc = nn.Sequential(
                nn.Linear(in_features=32 * 9 * 9, out_features=256),
                nn.ReLU(),
                nn.Linear(in_features=256, out_features=action_space)
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(128, 256),
                nn.LeakyReLU(),
                nn.Linear(256, 128),
                nn.LeakyReLU(),
                nn.Linear(128, action_space)
            )

    def forward(self, x):
        if self.format == 'image':
            x = self.conv(x).view(x.size()[0], -1)
        return self.fc(x)


class DQNAgent:
    def __init__(self,
                 observation_space,
                 action_space,
                 use_double_dqn=True,
                 lr=1e-4,
                 gamma=0.98,
                 device=torch.device("cuda" ),
                 eps=1.,
                 pretrained=None,
                 format = 'image'):

        self.use_double_dqn = use_double_dqn
        self.gamma = gamma
        self.eps = eps
        self.action_space = action_space
        self.policy_network = DQN(observation_space, action_space, format).to(device)
        self.target_network = DQN(observation_space, action_space, format).to(device)
        self.update_target_network()
        self.target_network.eval()

        if pretrained is not None:
            print(f"Loading a policy - {pretrained} ")
            print(self.policy_network.load_state_dict(torch.load(pretrained)))

        self.optimiser = torch.optim.Adam(self.policy_network.parameters(), lr=lr)

        self.device = device

    def update(self, transition_dict):
        """
        Optimise the TD-error over a single minibatch of transitions
        :return: the loss
        """
        b = len(transition_dict['states'])
        state = torch.tensor(transition_dict['states'], dtype=torch.float, device=self.device)
        action = torch.tensor(transition_dict['actions'], dtype=torch.int64).view(b, -1).to(self.device)
        reward = torch.tensor(transition_dict['rewards'],
                              dtype=torch.float).to(self.device)
        next_state = torch.tensor(transition_dict['next_states'], dtype=torch.float32,
                                  device=self.device)
        done = torch.tensor(transition_dict['dones'],
                            dtype=torch.float).to(self.device)

        with torch.no_grad():
            if self.use_double_dqn:
                _, max_next_action = self.policy_network(next_state).max(1)
                max_next_q_values = self.target_network(next_state).gather(1, max_next_action.unsqueeze(1)).squeeze()
            else:
                next_q_values = self.target_network(next_state)
                max_next_q_values, _ = next_q_values.max(1)
            target_q_values = reward + (1 - done) * self.gamma * max_next_q_values

        input_q_values = self.policy_network(state)
        input_q_values = input_q_values.gather(1, action).squeeze()

        loss = F.smooth_l1_loss(input_q_values, target_q_values)

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()


    def update_target_network(self):
        """
        Update the target Q-network by copying the weights from the current Q-network
        """
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def take_action(self, state):
        state = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze(0)
        if np.random.random() < self.eps:
            action = np.random.randint(self.action_space)
            return action
        with torch.no_grad():
            q_values = self.policy_network(state)
            _, action = q_values.max(1)
            return action.item()
