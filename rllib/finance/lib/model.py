import gym, ray
from ray.rllib.agents import ppo, dqn
from trading_gym_next import EnvParameter, TradingEnv
from gym import spaces
from backtesting.test import GOOG, EURUSD
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import random
from ray.rllib.models import ModelCatalog, ActionDistribution
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune.registry import register_env

#################################
#####         Net          ######
#################################

class SimpleFinanceModel(TorchModelV2):
    '''線形入力でDuelingNetworkを搭載したDQN'''
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        print(obs_space.shape[1], action_space.n)
        self.num_states = obs_space.shape[1]
        self.num_actions = action_space.n
        self.lstm = nn.LSTM(self.num_states, 32, batch_first=True)
        self.dense = nn.Linear(32, self.num_actions)

    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"] # (32, 10, 4) N, L, H
        x, _ = self.lstm(x)
        x = F.relu(x[:, -1, :])
        x = self.dense(x)
        return x, []
ModelCatalog.register_custom_model("simple_finance_model", SimpleFinanceModel)
