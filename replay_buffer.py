import random
import numpy as np
import collections

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.storage = collections.deque(maxlen=buffer_size)

    def add(self, state, action, reward, next_state, next_action):
        self.storage.append((state, action, reward, next_state, next_action))

    def sample(self, n):
        samples = random.sample(self.storage, n)

        states = torch.from_numpy(np)

