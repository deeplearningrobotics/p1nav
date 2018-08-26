# Banana Collector

## Environment Description

The environment is about training an agent to navigate a large square world wile
collection yellow banannas and avoiding blue ones.

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

* `0` - move forward.
* `1` - move backward.
* `2` - turn left.
* `3` - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

# Dependencies

```
pip3 install pytorch tensorboardX
```

To look at the graphs tensorboard is also required:
```
pip3 install tensorboard
```
# Running the Code

```
https://github.com/deeplearningrobotics/p1nav.git
git clone 
cd p1nav
python3 main.py
```

To monitor progress tensorboard can be used:
```
tensorboard --logdir runs
```


# 