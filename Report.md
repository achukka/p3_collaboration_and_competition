## Project 3: Collaboration and Competition

This is the third project as part of Deep Reinforcement Learning course in [Udacity](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893). The objective of the project is to train two agents to play tennis as long as possible.

---

### Problem Description

For this project we worked with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) , a two-player game where agents control rackets to bounce the ball over the net. Here the agents must bounce ball between one another while not dropping or sending ball out of bounds. 

The environment is considered sovled, when the average score (over 100 episodes) is atleast 0.5.

---

### Implementation Details

Since the above environment deals with multiple agents, I decided to solve it using Multi Agent Deep Deterministic Policy Gradient (MADDPG) introduced in this [paper](https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf). MADPPG is an extension to Deep Deterministic Agent ([DDPG](https://arxiv.org/pdf/1509.02971.pdf)) where each agent would have its own copy of critic and actor. However the actor is decentralized and critic is centralized, i.e actor can see only it's states but the critic can see all the agent's states as well as actions.

In the below image, the actors for each agents are represented in the red area, while critics are in green section.
![img](https://github.com/adityaharish/p3_collaboration_and_competition/blob/master/MADDPG.jpg)

Refer to the [implementation details](https://github.com/adityaharish/p2_continuous_control/blob/master/Report.md#implementation-details) in continuous control for more information on DDPG.

As in DDPG, the multi agent approach uses a common replay buffer by storing experience tuples and drawing sample independently during training.

#### Network Architecture

The network architecture remains almost same as previous, i.e 2 hidden layers each with `256` nodes. The actor takes in a state and outputs the action, where as the critic takes in (state+action) times the number of agents. We've also incorporated batch normalization in the first layer for both actor and critic.

The intial weights for the first two layers are intialized uniformly random using a `fan_in` mechanism. The final layers are though are clipped between `-3e3` and `3e3`. Instead of _ReLu_, we used _tanh_ for last year. As suggested from the Slack Channel, I kept both the intialization for both the actor and critics to be similar. This has helped in converging faster.

Changing the network architecture by tweaking things like number of hidden layers and their sizes didn't improve the performance.

---

### Results

I used [PyTorch](https://pytorch.org/) an open source deep learning platform by facebook to implement the neural network and Udacity AWS for GPU training. Refer to the [README](https://github.com/adityaharish/p2_continuous_control) for other run time dependencies. 

Note: [workspace_utils](https://github.com/adityaharish/p2_continuous_control/blob/master/workspace_utils.py) provided in the Udacity forums, helped me keep the workspace awake during the training.

#### Hyper Parameters

I did a fair bit of experimenting on the learning rates for both actor and critic models and found `1e-4` for actor and `1e-3` for the critic to be stable. Lower learning rates didn't took more episodes and high learning rates didn't speed up the proceedings.

For noise, I `Ornstein-Uhlenbeck process` as suggested in the previous exercise. I started with an intial amount `noise_start`(1.0) and decay it by `noise_decay`(0.9) until certain time steps `noise_max_timesteps`(300000) after which the noise is turned off. I've regulated the learning by `update_frequency`, so that agents will start learning only after every one got a chance to act. The soft update parameter `tau` can also be adapted, however I found `0.001` to work fine. 

One thing, I haven't quite played with is the buffer size. Increasing it might lead to faster convergence.

Following is the list of hyperparamters that we used during the training

| Parameter                   | Value  |
| ----------------------------|:------:|
| Epsiodes                    | 4000   |
| Epoch Size                  | 1000   |
| First Hidden Layer          | 256    |
| Second Hidden Layer         | 256    |
| Replay Buffer Size          | 10000 |
| Mini Batch Size             | 256 |
| Discount Rate               | 0.99 |
| Tau (Soft update parameter) | 0.001 |
| Learning Rate (Actor)       | 0.0001 |
| Learning Rate (Critic)      | 0.001 |
| Weight Decay                | 0.0001 |

You can find the models weights for both actors and critics in [Models](https://github.com/adityaharish/p3_collaboration_and_competition/tree/master/Models)


#### Observations

The environment was sovled in *2998* epsiodes with an average score of `0.518`. As seen from the picture below, the learning was too slow in the beginning and picked up during the later stages. 

![img](https://github.com/adityaharish/p3_collaboration_and_competition/blob/master/Scores.png)

As expected, the collaborative aspect makes the training rough and longer. I suspect that the noise factor accounts for irregular bumps in average scores at the middle. However during the later stages (where there is no noise), I see per episode scores reaching as high as `3`, but stil the average score doesn't add upto THRESHOLD. Increasing the buffer size and batch size might help reducing the total number of episdoes, but that could be a problem in larger setting (games with more agents). We should find an effective way to communicate information between the agents.

---

### Further Improvements

- Instead of Replay Buffer we could try Prioritized Replay ([paper](https://arxiv.org/abs/1511.05952)).
- Other algorithms like Proximal Policy Optimization could potentially lead to better results since the dimension is not quite high.
- We could try [Q-prop](https://arxiv.org/abs/1611.02247) algorithm combines both off-policy and on-policy learning.
- Optimization techniques like dropouts, early stopping and warm restarts could reduce the training time.
- Better maintenance of noise might lead to faster convergence.
- As mentioned previously, increasing buffer size and batch size might help.    
