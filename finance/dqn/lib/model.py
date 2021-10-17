from abc import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#################################
#####         Net          ######
#################################

class SimpleFinanceNet(nn.Module):
    '''線形入力でDuelingNetworkを搭載したDQN'''
    def __init__(self, num_states, num_actions):
        super(SimpleFinanceNet, self).__init__()
        self.num_states = num_states
        self.num_actions = num_actions
        self.lstm = nn.LSTM(num_states, 32)
        self.dense = nn.Linear(32, num_actions)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = F.relu(x[-1, :, :])
        x = self.dense(x)
        return x


class SimpleFinanceNetWithPosition(nn.Module):
    '''線形入力でDuelingNetworkを搭載したDQN'''
    def __init__(self, num_states, num_actions):
        super(SimpleFinanceNetWithPosition, self).__init__()
        self.num_states = num_states
        self.num_actions = num_actions
        self.lstm = nn.LSTM(num_states["market"], 32)
        self.dense = nn.Linear(num_states["position"], 32)
        # concat market(32) and position(32)
        self.concat = torch.cat([])
        self.fc1 = nn.Linear(64, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))

        V = self.fcV2(self.fcV1(x))
        A = self.fcA2(self.fcA1(x))

        averageA = A.mean(1).unsqueeze(1)
        return V.expand(-1, self.num_actions) + (A - averageA.expand(-1, self.num_actions))


class AdvantageFinanceNet(nn.Module):
    '''線形入力でDuelingNetworkを搭載したDQN'''
    def __init__(self, num_states, num_actions):
        super(AdvantageFinanceNet, self).__init__()
        self.num_states = num_states
        self.num_actions = num_actions
        self.lstm = nn.LSTM(num_states, 32)
        self.fc1 = nn.Linear(32, 32)

        self.fcV1 = nn.Linear(32, 32)
        self.fcA1 = nn.Linear(32, 32)
        self.fcV2 = nn.Linear(32, 1)
        self.fcA2 = nn.Linear(32, self.num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))

        V = self.fcV2(self.fcV1(x))
        A = self.fcA2(self.fcA1(x))

        averageA = A.mean(1).unsqueeze(1)
        return V.expand(-1, self.num_actions) + (A - averageA.expand(-1, self.num_actions))


class AdvantageFinanceNetWithPosition(nn.Module):
    '''線形入力でDuelingNetworkを搭載したDQN'''
    def __init__(self, num_states, num_actions):
        super(AdvantageFinanceNetWithPosition, self).__init__()
        self.num_states = num_states
        self.num_actions = num_actions
        self.lstm = nn.LSTM(num_states, 32)
        self.dense = nn.Linear(num_positions, 32)
        # concat market(32) and position(32)
        self.concat = torch.cat([])
        self.fc1 = nn.Linear(64, 64)

        self.fcV1 = nn.Linear(64, 64)
        self.fcA1 = nn.Linear(64, 64)
        self.fcV2 = nn.Linear(64, 1)
        self.fcA2 = nn.Linear(64, self.num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))

        V = self.fcV2(self.fcV1(x))
        A = self.fcA2(self.fcA1(x))

        averageA = A.mean(1).unsqueeze(1)
        return V.expand(-1, self.num_actions) + (A - averageA.expand(-1, self.num_actions))
