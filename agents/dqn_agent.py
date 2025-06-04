# agents/dqn_agent.py
from networks.q_network import QNetwork
from utils.replay_buffer import ReplayBuffer
import torch
import torch.nn.functional as F
import numpy as np
import random
import copy
#The DQN Agent that will be used to solve LundaLander-v3 environment and can optionally use DOuble DQN
class DQNAgent:
    def __init__(self, env, use_double_dqn: bool = False):
        self.env = env #setting the gymnasium environment
        self.use_double_dqn = use_double_dqn #boolean to set if Double Dqn logic will be used
        self.state_size = env.observation_space.shape[0] #Size of the observation space
        self.action_size = env.action_space.n #Numver of discrete actions available

        self.q_net = QNetwork(self.state_size, self.action_size) #the online Q network
        self.target_net = copy.deepcopy(self.q_net) #target Q network
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=1e-3) #optimizer for training Q network
        self.buffer = ReplayBuffer(100000) #replay buffer for storing experiences
        self.batch_size = 64 #number of transitions per training step
        self.gamma = 0.99 #Discount factor for future rewards
        self.epsilon = 1.0 #epsilon greedy exploration rate
        self.target_update_counter = 0 #tracks when to update the target number

    def select_action(self, state):
        """
        Select an action using epsilon-greedy strategy.

        :param:
            state: The current environment state.

        :return:
            The selected action (int).
        """
        if random.random() < self.epsilon:
            return self.env.action_space.sample() #explore part
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax().item() #exploit part

    def store_transition(self, s, a, r, s_, done):
        """
        Store a transition in the replay buffer.

        :param:
            s: Current state.
            a: Action taken.
            r: Reward received.
            s_: Next state.
            done: Whether the episode ended.
        """
        self.buffer.add(s, a, r, s_, done)

    def train(self):
        """
        Perform one training step using a batch of experiences from the replay buffer.
        """
        # if not enough data to sample a batch
        if len(self.buffer) < self.batch_size:
            return
        #sample a batch of transitions form the replay buffer
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        #convert to tensor
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        #current q values for the selected actions
        q_values = self.q_net(states).gather(1, actions)

        #compute the target Q values using either DDouble DQN or regular DQN
        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN: decouple action selection and evaluation
                next_actions = self.q_net(next_states).argmax(1, keepdim=True)
                target_q_values = self.target_net(next_states).gather(1, next_actions)
            else:
                # regular DQN
                target_q_values = self.target_net(next_states).max(1, keepdim=True)[0]
            #compute the target values
            targets = rewards + self.gamma * target_q_values * (1 - dones)

        #compute loss between predicted and target Q-values
        loss = F.mse_loss(q_values, targets)
        #ompitimize the Q-network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        #set the epsilon for exploration - exploitation tradeoff
        self.epsilon = max(0.01, self.epsilon * 0.995)

        # Update target network every 100 steps
        self.target_update_counter += 1
        if self.target_update_counter % 100 == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

