import numpy as np
import random
from collections import namedtuple, deque
from operator import itemgetter

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e4)  # replay buffer size
BATCH_SIZE = 32         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-5               # learning rate
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        indices, states, actions, rewards, next_states, dones, prob_samples = experiences

        beta = 0.4
        weights = torch.pow(len(self.memory)*prob_samples, -beta)
        max_weight = torch.max(weights)
        weights = weights/max_weight
        weights = weights.unsqueeze(-1).detach()


        # Get max predicted Q values (for next states) from target model
        best_actions = self.qnetwork_local(next_states).detach().argmax(1)
        best_actions = best_actions.unsqueeze(-1)
        Q_targets_next = self.qnetwork_target(next_states).detach().gather(1, best_actions)
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        updated_priorities = torch.abs(Q_targets)
        self.memory.update_priorities(indices, updated_priorities)

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(weights*Q_expected, weights*Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, alpha = 0.9):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.memory = deque(maxlen=self.buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.alpha = alpha
        self.max_priority = 1.0
        self.max_priority_current = False
        self.priorities = deque(maxlen=self.buffer_size)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        if not self.max_priority_current and len(self.priorities) > 1:
            self.max_priority = max(self.priorities)
            self.max_priority_current = True

        self.priorities.append(self.max_priority)
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        alpha = 0.6
        prio_alpha = np.power(self.priorities, alpha)
        prob = np.asarray(prio_alpha)/np.sum(prio_alpha)
        indices = np.random.choice(range(0, len(self.priorities)), size=self.batch_size, p=prob)

        assert(len(self.priorities) == len(self.memory))
        assert(len(indices) == self.batch_size)
        experiences = itemgetter(*indices)(self.memory)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        prob_samples = torch.from_numpy(prob[indices]).float().to(device)

        return indices, states, actions, rewards, next_states, dones, prob_samples

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    def update_priorities(self, indices, updated_priorities):
        self.max_priority_current = False

        pa = np.asarray(self.priorities).squeeze()
        pa[indices] = np.asarray(updated_priorities).squeeze()
        self.priorities = deque(pa.tolist(), maxlen=self.buffer_size)

