import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import OrderedDict

class Agent:
    def __init__(self, obs_size, action_size, epsilon_greedy=0.05):
        self.obs_size = obs_size
        self.action_size = action_size
        self.net = nn.Sequential(OrderedDict([('fc1', nn.Linear(obs_size, 64)),
                                             ('r1', nn.ReLU()),
                                             ('fc2', nn.Linear(64, action_size))]))

        self.epsilon_greedy = epsilon_greedy

    def Q(self, obs, action):
        return self.net.forward(obs)[0][action]

    def act(self, obs):
        if random.random() < self.epsilon_greedy:
            return np.asarray([random.randint(0, self.action_size-1)])
        else:
            obs = torch.from_numpy(obs).float()
            return np.argmax(self.net.forward(obs).detach().numpy())


