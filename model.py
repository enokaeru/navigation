import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetworklow(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64, hiddens=256):
        """
         Initialize parameters and build model for low-input
         (ref:https://github.com/openai/baselines/blob/master/baselines/deepq/models.py)


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
        self.fc3 = nn.Linear(fc2_units, hiddens)

        self.action_out = nn.Linear(hiddens, action_size)
        self.state_out = nn.Linear(hiddens, state_size)
        self.state_score = nn.Linear(state_size, 1)


    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        fc_out = x.squeeze()
        state_out = F.relu(self.state_out(fc_out))
        action_out = F.relu(self.action_out(fc_out))
        state_scores = self.state_score(state_out)
        action_scores_mean = action_out.mean()
        action_score_centered = action_out - action_scores_mean.expand_as(action_out)
        return state_scores + action_score_centered
