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
#####         Env          ######
#################################

class FinanceEnv(gym.Wrapper):
    def __init__(self, env_config):
        param = env_config["param"]
        param.window_size += 1 # because drop by log_diff
        super().__init__(TradingEnv(param))
        self.action_space = spaces.Discrete(3) # action is [0, 1, 2] 0: HOLD, 1: BUY, 2: SELL
        self.observation_space = spaces.Box(low=-10, high=10, shape=(param.window_size-1, 4), dtype=np.float64)
        self.position_side = 0 # 0: No Position, 1: Long Position, 2: Short Position
        self.step_num = 0

    def step(self, action):
        if self.position_side == 1 and action == 1:
            action = 0
        if self.position_side == 2 and action == 2:
            action = 0
        
        if self.position_side != 0:
            self.step_num += 1
        if self.step_num >= 20:
            if self.position_side == 1:
                action = 2
            elif self.position_side == 2:
                action = 1

        obs, reward, done, info = self.env.step(action, size=1)
        logdiff = self.log_diff(obs.loc[:,['Close','Volume']])
        obs = pd.merge(logdiff, obs.loc[:,['Position','Reward']], left_index=True, right_index=True)
        if (self.position_side == 1 and action == 2) or (self.position_side == 2 and action == 1) or self.step_num >= 20:
            done = True

        if info["position"].size > 0:
            self.position_side = 1
        elif info["position"].size < 0:
            self.position_side = 2
        return obs.to_numpy(), reward, done, info
    
    def log_diff(self, df):
        return np.log(df).diff(1)[1:]

    def reset(self):
        self.step_num = 0
        self.position_side = 0
        obs = self.env.reset()
        logdiff = self.log_diff(obs.loc[:,['Close','Volume']])
        obs = pd.merge(logdiff, obs.loc[:,['Position','Reward']], left_index=True, right_index=True)
        return obs.to_numpy()
