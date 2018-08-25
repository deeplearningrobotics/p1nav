# Report

## Algorithm

Classical DQN is used as described here: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
The single derivation is the use of a target network as described
here https://arxiv.org/pdf/1509.06461.pdf (3).

### Hyperparameters

Replay buffer size: 10000
Minibatch size: 64
Discount factor: 0.99
Soft update of target parameters: 1e-3
Learning rate: 5e-4
Number of network updates: 4
