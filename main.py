from unityagents import UnityEnvironment
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

device = torch.device("cuda:0")
print(torch.cuda.is_available())

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

env = UnityEnvironment(file_name="Banana_Linux/Banana.x86_64", no_graphics=True)
brain_name = env.brain_names[0]


from dqn_agent import Agent

agent = Agent(state_size=37, action_size=4, seed=0)
writer = SummaryWriter()

def dqn(n_episodes=500, max_t=1000, eps_start=1.0, eps_end=0.00, eps_decay=0.995):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon
    for i_episode in range(1, n_episodes + 1):
        state = env.reset()[brain_name].vector_observations
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            # next_state, reward, done, _ = env.step(action)
            brain_info = env.step(action)[brain_name]
            next_state = brain_info.vector_observations
            reward = brain_info.rewards
            done = any(brain_info.local_done)

            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += sum(reward)
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        writer.add_scalar('reward', np.mean(scores_window), i_episode)
        writer.add_scalar('eps', eps, i_episode)
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), str(np.mean(scores_window))+'.pth')
    return scores


scores = dqn()
