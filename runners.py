from agents import Agent
from replay_buffer import ReplayBuffer
import torch
import torch.nn as nn
import torch.nn.functional as F
class Runner:
    def __init__(self, env):

        self.env = env
        brain_name = self.env.brain_names[0]
        brain = self.env.brains[brain_name]

        env_info = self.env.reset(train_mode=True)[brain_name]

        self.action_size = brain.vector_action_space_size

        state = env_info.vector_observations[0]
        self.state_size = len(state)

        self.buffer = ReplayBuffer(10000)
        self.agent = Agent(self.state_size, self.action_size)

        self.brain_name = self.env.brain_names[0]

        self.steps = 1000
        self.discount = 1

        self.optimizer = torch.optim.Adam(self.agent.net.parameters(), lr=1e-2)

    def run(self):

        while True:
            ep_rewards = 0
            done = False
            brain_info = self.env.reset(train_mode=True)[self.brain_name]
            #print(brain_info.vector_observations)
            action = self.agent.act(brain_info.vector_observations)
            for _ in range(0, self.steps):
                brain_info = self.env.step(action)[self.brain_name]
                next_obs = brain_info.vector_observations
                reward = brain_info.rewards
                done = any(brain_info.local_done)

                next_action = self.agent.act(next_obs)

                self.buffer.add(brain_info.vector_observations, action, reward, next_obs, next_action)

                obs = next_obs
                action = next_action

                ep_rewards += sum(reward)
                if done:
                    brain_info = self.env.reset(train_mode=True)[self.brain_name]
                    action = self.agent.act(brain_info.vector_observations)

            print(ep_rewards)

            self.train()

    def train(self):
        sampled_rew = 0
        samples = self.buffer.sample(256)
        for sample in samples:
            obs, action, reward, next_obs, next_action = sample
            # #print(reward)
            # y = reward + self.discount*self.agent.act(next_obs)
            # y = torch.from_numpy(y).float()
            #
            # obs = torch.from_numpy(obs).float()
            # qjj = self.agent.Q(obs, action)

            Q_targets_next = self.agent.net(next_obs).detach().max(1)[0].unsqueeze(1)
            # Compute Q targets for current states
            Q_targets = rewards + (0.99 * Q_targets_next)

            # Get expected Q values from local model
            Q_expected = self.qnetwork_local(states).gather(1, actions)

            loss = F.mse_loss(y, qjj)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            sampled_rew += sum(reward)

        print("Sampled rew:" + str(sampled_rew))







