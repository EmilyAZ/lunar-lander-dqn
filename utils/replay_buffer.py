# utils/replay_buffer.py
import random
from collections import deque
import numpy as np

class ReplayBuffer:
    """fixed sized buffer used to store experience tuples for training of LunarLander"""
    def __init__(self, capacity: int):
        """
        Initialized the replay buffer
        :param capacity: max number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        """
        add a single transition to the replay buffer

        :param state: current state
        :param action: action taken
        :param reward: reward received
        :param next_state: next state after taking the action
        :param done: whether the episode ended after this step
        :return:
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        """
        samples a random batch of transitions from the buffer
        :param batch_size: number of transitions to sample
        :return: tuple of arrays
        """
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        """
        returns the length of the current buffer
        :return: number of elements in buffer
        """
        return len(self.buffer)
