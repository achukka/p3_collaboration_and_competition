## Project 3: Collaboration and Competition

This is the third project as part of Deep Reinforcement Learning course in [Udacity](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893). The objective of the project is to train two agents to play tennis as long as posibble.


---

### Problem Description

For this project we worked with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis)
, a two-player game where agents control rackets to bounce the ball over the net. Here the agents must bounce ball between one another while not dropping
or sendig ball out of bounds. 

The environment is considered sovled, when the average score (over 100 episodes) is atleast 0.5.

---

### Implementation Details
<!--
At the heart of this approach lies an actor-critic method. Policy-gradient methods like [REINFORCE](http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf) use Monte-Carlo esitmate.
As a result they exhibit high variance. Value-based approaches using Temporal Difference estimates display low variance. Actor-Critic methods combine these two
approaches and extract the best out of both worlds. They are more stable than TD estimates and at the same time need fewer samples than policy-gradient methods.

An Actor-Critic method contains two neural network one for the actor and one for the critic. The actor's role is to update the policy which is then evaluted by the critc, in turn training the actor to acheive a good policy. 

In order to update the policy in vanila Policy-gradient methods we do the following 
- Accumulate rewards over the episode.
- Compute the average them.
- Calculate the gradient.
- Perform gradient descent and update the policy.

![img](https://github.com/adityaharish/p2_continuous_control/blob/master/images/policy_gradient_update.gif)
&nbsp;

In Actor-Critic methods we use the value provided by the critic to update the actors policy.

![img](https://github.com/adityaharish/p2_continuous_control/blob/master/images/actor_critic_update.gif)
&nbsp;

[Deep Deterministic Policy Gradient (DDPG)](https://arxiv.org/pdf/1509.02971.pdf) is the version of actor-critic method we used to solve the above environment. Here the actor generates a deterministic policy which is evaluated by the critic. Note that some of the actor-critic variants use stochastic policies. We update the critic using TD error and action learns using the following deterministic policy gradient.
&nbsp;

![img](https://github.com/adityaharish/p2_continuous_control/blob/master/images/DDPG%20update%20equation.gif)

In the above equation the actor is represented using a parametrized function <img src="https://github.com/adityaharish/p2_continuous_control/blob/master/images/actor_policy.gif"/> and `Q(s,a)` is the critic.

The above update step is ran for each agent (in this case 20)at regular intervals. There are also a few more techniques that were incorporated to stabilize the training.

#### Fixed Targets
This was introduced as part of Deep Q Network that proved to be quite useful in learning and updating weights.
Both the actor and critic would have two neural neworks (a local and a target). As with DQN we freeze the target network and learn weights for the local model. 

#### Soft udpates
In DQN, we update the target weights after every `C` steps. In DDPG we update for every step but in a soft manner, i.e 
```math w_t = 99.99% * w_t + 0.01% w_l ``` where `w_t` is weight of target and `w_l` is weight of local. This way most of the target network weights are retained and moved slowly.

#### Experience Replay
In most of the RL techniques the adjacent states (or experience tuples) are interlinked. So our model might get biased to a certain behavior (or path) and never learn other paths. Hence it is highly necessary that our samples are independent. To acheive this purpose, we maintain a Replay Buffer of experience tuples. After a fixed number of iterations we randomly sample an experience from this buffer, calcualte the loss and eventually update the parameters. This way, we break correlations between adjacent tuples and stabilize the learning. Around the same time, it helps us re-use the experience instead of running through the environment again.
-->
---

### Results

I used [PyTorch](https://pytorch.org/) an open source deep learning platform by facebook to implement the neural network and Udacity AWS for GPU training. Refer to the [README](https://github.com/adityaharish/p2_continuous_control) for other run time dependencies. 

The Slack Channel suggested few ideas on improving the model and one of them was to use Batch Normalization. I have used it at the first layer in both the actor and critic which enhanced the learning process. Leaky ReLu (with `0.01` slope) proved to work much better than ReLu. 

Note: [workspace_utils](https://github.com/adityaharish/p2_continuous_control/blob/master/workspace_utils.py) provided in the Udacity forums, helped me keep the workspace awake during the training.

#### Hyper Parameters

Following is the list of hyperparamters that we used during the training

| Parameter                   | Value  |
| ----------------------------|:------:|
| Epoch Size epsiodes         | 1000   |
| First Hidden Layer          | 256    |
| Second Hidden Layer         | 128    |
| Third Hidden Layer (Critic) | 128    |
| Leaky ReLu (Slope)          | 0.01 |
| Replay Buffer Size          | 1000000 |
| Mini Batch Size             | 1024 |
| Discount Rate               | 0.99 |
| Tau (Soft update parameter) | 0.001 |
| Learning Rate (Actor)       | 0.0001 |
| Learning Rate (Critic)      | 0.0003 |
| Weight Decay                | 0.0001 |

<!--
You can find the models weights for [actor](https://github.com/adityaharish/p2_continuous_control/blob/master/Results/checkpoint_actor.pth) and [critic](https://github.com/adityaharish/p2_continuous_control/blob/master/Results/checkpoint_critic.pth) in the Results folder.
-->

#### Observations
<!--
The environment was sovled in *318* epsiodes with an average score of `30.07`. The learning was too slow in the beginning and picked up during the later stages. However DDPG updates seem to much smoother than the earlier approaches we've played with as evident from the graph below.

![img](https://github.com/adityaharish/p2_continuous_control/blob/master/Results/ddpg_scores.png)

Batch Normalization could be one of the reason for smoothness in DDPG. Increasing Epoch size for each episode didn't alter the performance much. Also altering the learning rates didn't influence the training much.
-->

---

### Further Improvements
<!--
- Using Prioritized Replay ([paper](https://arxiv.org/abs/1511.05952)) has generally shown to have been quite useful. So we could give that one a try.
- Other algorithms like TRPO, PPO, A3C, A2C that have been discussed in the course could potentially lead to better results as well.
- We could try [Q-prop](https://arxiv.org/abs/1611.02247) algorithm combines both off-policy and on-policy learning.
- Clipping gradients and (hard) setting both critic and target to same set of weights might yield a better performance.
- Optimization techniques like dropouts, early stopping and warm restarts could reduce the training time.
-->
