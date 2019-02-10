from torch.optim import Adam
import torch
import torch.nn.functional as F
import numpy as np

from collections import deque, namedtuple
import random
from model import Actor_Critics
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MADDPG(object):
    """Meta agent that contains `n_agent` ddpg agents"""
    
    def __init__(self, state_size=24, 
                 action_size=2, 
                 seed=0,
                 n_agents=2,
                 buffer_size=10000,
                 batch_size=256,
                 gamma = 0.99,
                 update_frequency=2,
                 noise_start=1.0,
                 noise_decay=0.9,
                 noise_max_timesteps=30000):
        """
        Parameters
        ==========
            state_size (int): size of each state
            action_size (int): size of each action
            seed (int): Random seed
            n_agents (int): number of agents
            buffer_size (int): Size of the replay buffer
            batch_size (int): Size of mini batch            
            gamma (float): discount factor
            update_frequency (int): frequency of updates
            noise_start (float): starting noise rate
            noise_decay (float): decay rate for noise
            noise_max_timesteps (int): maximum timesteps for noise in training
        """
        super(MADDPG, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.update_frequency = update_frequency
        self.noise_wt = noise_start
        self.noise_decay = noise_decay        
        
        self.noise_max_timesteps = noise_max_timesteps
        self.add_noise = True
        self.noise_timestep = 0

        # Create agents, with actor and critic
        models = [Actor_Critics(n_agents, state_size, action_size, seed=seed) for _ in range(n_agents)]
        self.agents = [DDPG(i, models[i], action_size,  seed) for i in range(n_agents)]
        
        # Shared Replay buffers
        self.replay_buffer = ReplayBuffer(buffer_size, batch_size, seed)

    def act(self, states_all_agents, add_noise=True):        
        """get actions from all agents in the MADDPG object"""
        actions_all_agents = []
        for agent, state in zip(self.agents, states_all_agents):
            actions_all_agents.append(agent.act(state, noise_wt=self.noise_wt, add_noise=self.add_noise))
            self.noise_wt *= self.noise_decay
        return np.array(actions_all_agents).reshape(1, -1) # Reshape into row vector (1 x 4)

    def learn(self, experiences, gamma):
        # Calculate action for each agent using its actor
        next_actions_all_agents = []
        actions_all_agents = []
        
        for experiencetups, agent in zip(experiences, self.agents):
            states, _, _, next_states, _ = experiencetups
            
            # Get agent's state and action using actor local network
            state = states.reshape(-1, 2, self.state_size).index_select(1, agent.agent_id).squeeze(1)
            action = agent.actor_local(state)
            actions_all_agents.append(action)

            # Get agent's next state and next action using target network
            next_state = next_states.reshape(-1, 2, self.state_size).index_select(1, agent.agent_id).squeeze(1)
            next_action = agent.actor_target(next_state)
            next_actions_all_agents.append(next_action)            
            
        # Learn for each agent using experience sample
        for experience, agent in zip(experiences, self.agents):
            agent.learn(experience, gamma, next_actions_all_agents, actions_all_agents)


    def save_agents(self):
        """Save Models for each agent (actor+critic)"""
        for agent in self.agents:
            torch.save(agent.actor_local.state_dict(), f"checkpoint_actor_agent_{agent.agent_id_int}.pth")            
            torch.save(agent.critic_local.state_dict(), f"checkpoint_critic_agent_{agent.agent_id_int}.pth")


    def step(self, states_all_agents, actions_all_agents, rewards_all_agents, next_states_all_agents, dones_all_agents):
        states_all_agents = states_all_agents.reshape(1, -1) # Reshape into row vector (1 x 48)
        next_states_all_agents = next_states_all_agents.reshape(1, -1) # Reshape into row vector (1 x 48)

        self.replay_buffer.add(states_all_agents, actions_all_agents, rewards_all_agents, next_states_all_agents, dones_all_agents)

        # Turn off the noise after noise_max_timesteps
        if self.noise_timestep > self.noise_max_timesteps:
            self.add_noise = False

        self.noise_timestep += 1
        # Learn according to the update_frequency and if enough samples are avaialable
        if self.noise_timestep % self.update_frequency == 0 and len(self.replay_buffer) > self.batch_size:            
            experiences = [self.replay_buffer.sample() for _ in range(len(self.agents))]
            self.learn(experiences, self.gamma)

                
class DDPG(object):
    """DDPG Agent with an actor critic"""
    def __init__(self, agent_id, model, action_size,  seed, tau=0.001, lr_actor=1e-4, lr_critic=1e-3, weight_decay=0.0):
        """Initialize parameters and build DDPG actor and critic.
        ======
            agent_id (int): Id of each agent
            model: model object
            action_size (int): size of each action            
            seed (int): Random seed
            tau (float): for soft update of target parameters
            lr_actor (float): learning rate for actor
            lr_critic (float): learning rate for critic
            weight_decay (float): L2 weight decay
        """
        
        random.seed(seed)
        self.agent_id = torch.tensor([agent_id]).to(device)
        self.agent_id_int = agent_id
        self.model = model
        self.action_size = action_size
        self.tau = tau
        self.lr_actor = lr_actor
        self.lr_critic= lr_critic
        
        # Network for Actor
        self.actor_local = model.actor_local
        self.actor_target = model.actor_target
        self.actor_optimizer = Adam(self.actor_local.parameters(), lr=lr_actor)
        
        # Critic Network
        self.critic_local = model.critic_local
        self.critic_target = model.critic_target
        self.critic_optimizer = Adam(self.critic_local.parameters(), lr=lr_critic, weight_decay=weight_decay)
        
        # Copy weights from local network to target network
        self.hard_update(self.actor_local, self.actor_target)
        self.hard_update(self.critic_local, self.critic_target)
                
        # Noise process
        self.noise = OUNoise(action_size, seed)    

    def hard_update(self, source, target):
        """
        Copy network parameters from source to target
        Inputs:
            target (torch.nn.Module): Net to copy parameters to
            source (torch.nn.Module): Net whose parameters to copy
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

        
    def act(self, state, noise_wt=1.0, add_noise=True):
        """Compute action for the given state based on the policy """
        state = torch.from_numpy(state).float().to(device)
        # Evaluate local actions
        self.actor_local.eval()
        
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        
        if add_noise:
            noise_val = self.noise.sample() * noise_wt
            action += noise_val
        return np.clip(action, -1, 1)

    
    def reset(self):
        """Reset noise"""
        self.noise.reset()
        
        
    def learn(self, experiences, gamma, next_actions_all_agents, actions_all_agents):
        """Update policy and value params using a set of experience tuples"""
        states, actions, rewards, next_states, dones = experiences
                
        """ --- Update critic ---- """
        self.critic_optimizer.zero_grad()        
        next_actions = torch.cat(next_actions_all_agents, dim=1).to(device)
        
        with torch.no_grad():
            q_targets_next = self.critic_target(next_states, next_actions)
                
        # Compute Q targets for current states
        q_expected = self.critic_local(states, actions)
        # Q_t = r_t + gamma * Q_{t+1}
        q_targets = rewards.index_select(1, self.agent_id) + (gamma * q_targets_next * (1 - dones.index_select(1, self.agent_id)))
        
        # Compute  Loss
        critic_loss = F.mse_loss(q_expected, q_targets.detach())        
        # Minimize loss
        critic_loss.backward()
        self.critic_optimizer.step()
        
        """ --- Update Actor ---- """
        self.actor_optimizer.zero_grad()
        # collect actions for this agent
        predicted_actions = [actions if index == self.agent_id_int else actions.detach() for index, actions in enumerate(actions_all_agents)]
        # Create a tensor
        predicted_actions = torch.cat(predicted_actions, dim=1).to(device)
        
        # Compute loss 
        actor_loss = -self.critic_local(states, predicted_actions).mean()
        # Minimize loss
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update both critic and actor
        self.soft_update(self.critic_local, self.critic_target, tau=self.tau)
        self.soft_update(self.actor_local, self.actor_target, tau=self.tau)
    
    def soft_update(self, source, target, tau=1e-3):
        """Move target params toward source based on weight    
        theta_target = tau * theta_source + (1 - tau) * theta_target
        """
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * source_param + (1.0 - tau) * target_param.data)
            

class OUNoise(object):
    """Ornstein-Uhlenbeck process."""
    
    def __init__(self, action_size, seed, scale=0.1, mu=0.0, theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        random.seed(seed)
        np.random.seed(seed)
        self.action_size = action_size
        self.scale = scale
        self.mu = mu * np.ones(self.action_size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset state to mean """
        self.state = copy.copy(self.mu)

    def sample(self):
        """Sample noise and add it to state"""        
        dstate = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_size)
        self.state += dstate
        return self.state  


class ReplayBuffer(object):
    """Fixed size buffer to store experiences"""
    
    def __init__(self, size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): Random seed
        """      
        random.seed(seed)
        np.random.seed(seed)
        self.size = size
        self.batch_size = batch_size
        self.buffer = deque(maxlen=self.size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add experience into the buffer"""
        exp = self.experience(state, action, reward, next_state, done)
        self.buffer.append(exp)
            
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.buffer, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.buffer)