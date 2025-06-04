# networks/q_network.py
import torch.nn as nn

class QNetwork(nn.Module):
    """
    A fully connected Q-Network for approximating the Q-value function.

    :param:
        state_size (int): The size of the input state vector.
        action_size (int): The number of discrete actions available.
    """
    def __init__(self, state_size: int, action_size: int):
        super(QNetwork, self).__init__()

        #Defines a 3 layer fully connected network
        self.fc = nn.Sequential(
            nn.Linear(state_size, 128), #input layer with 128 hidden units
            nn.ReLU(),
            nn.Linear(128, 128), #hidden layer with 128 hidden units
            nn.ReLU(),
            nn.Linear(128, action_size) #output layer
        )

    def forward(self, state):
        """
        Forward pass to compute Q-values from input state.

        :param:
            state (Tensor): A batch of states with the shape (batch_size, state_size).

        :return:
            Tensor: A tensor of Q-values with the shape (batch_size, action_size).
        """
        return self.fc(state)
