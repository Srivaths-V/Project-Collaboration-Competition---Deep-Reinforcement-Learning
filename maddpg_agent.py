import copy
import os
import random
from collections import namedtuple, deque

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from ddpg_agent import ddpg_agent, ReplayBuffer
from model import Actor, Critic

BUFFER_SIZE = int(1e6)          # Size of Replay Buffer
BATCH_SIZE = 256               # Size of minibatch
LEARNING_INTERVAL = 2             # Regularity of weight update

# PyTorch device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MADDPG_Agent:
    """Class for all agents in the environment."""

    def __init__(self, num_agents=2, state_size=24, action_size=2):
        """Initialize a MADDPG_agent object.
        Params
        ======
            num_agents (int): the number of agents in the environment
            state_size (int): dimension of each state
            action_size (int): dimension of each action
        """
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        
        self.agents = [ddpg_agent(state_size, action_size, i+1, random_seed=0) for i in range(num_agents)]
        
        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed=0)
        
    def reset(self):
        """Resets OU Noise for each agent."""
        for agent in self.agents:
            agent.reset()
            
    def act(self, observations, add_noise=True):
        """"Get actions from all agents"""
        
        actions = [agent.act(observation,add_noise=noise) for agent, observation in zip(self.agents, observations_all_agents)]
        return np.array(actions)
    
    def step(self, states, actions, rewards, next_states, dones, timestep):
        """Save experience in replay memory."""
        
        states = states.reshape(1, -1)
        actions = actions.reshape(1, -1)
        next_states = next_states.reshape(1, -1)
        
        self.memory.add(states, actions, rewards, next_states, dones)
        
        if len(self.memory) > BATCH_SIZE and timestep % LEARNING_INTERVAL == 0:
            for agent_id, agent in enumerate(self.agents):
                experiences = self.memory.sample()
                self.learn(experiences, agent_id)
            
    def learn(self, experiences, agent_number):
        """ The critic takes the input of the combined observations and 
        actions from all agents. Collect actions from each agent for the 'experiences'. """
        next_actions = []
        actions_pred = []
        states, _, _, next_states, _ = experiences
        
        next_states = next_states.reshape(-1, self.num_agents, self.state_size)
        states = states.reshape(-1, self.num_agents, self.state_size)
        
        for agent_id, agent in enumerate(self.agents):
            
            agent_id_tensor = torch.tensor([agent_id]).to(device)
            
            state = states.index_select(1, agent_id_tensor).squeeze(1)
            next_state = next_states.index_select(1, agent_id_tensor).squeeze(1)
            
            next_actions.append(agent.actor_target(next_state))
            actions_pred.append(agent.actor_local(state))
            
        next_actions = torch.cat(next_actions, dim=1).to(device)
        actions_pred = torch.cat(actions_pred, dim=1).to(device)
        
        agent = self.agents[agent_number]
        agent.learn(experiences, next_actions, actions_pred)
    