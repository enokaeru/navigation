import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetworklow(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """
         Initialize parameters and build model for low-input

         Parameters
         ----------
         stat_size : int
         action_size: int
         seed: int
        """
        super(QNetworklow, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
