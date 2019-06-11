import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetworklow(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
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
        self.fc3 = nn.Linear(fc2_units, action_size)

        self.state_out = nn.Linear(fc2_units, state_size)
        self.state_score = nn.Linear(state_size, 1)


    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        state_score = F.relu(self.state_out(x))
        action_scores = self.fc3(x)
        action_scores_mean = action_scores.mean()
        action_score_centered = action_scores - action_scores_mean.expend()
        return state_score + action_score_centered
