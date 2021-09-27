from abc import *
import torch
import gym
from typing import NamedTuple
import torchvision.transforms as T
import numpy as np
from gym import spaces
from gym.spaces.box import Box
from PIL import Image
from matplotlib import pyplot as plt

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#################################
#####  Stock Market Env  ######
#################################

class EnvParameter(NamedTuple):
    max_lot: int    # 最大数量
    spread: int     # スプレッド
    window_size: int
        
class StockMarketEnv(gym.Wrapper):
    def __init__(self, df):
        self.df = df
        self.action_space = spaces.Discrete(3) # action is [0, 1, 2] 0: HOLD, 1: SELL, 2: BUY
        
    
    def reset(self):
        observation = self.env.reset() # (210, 160, 3) (h,w,c)

        # fire action at first
        fire_action = 3
        observation, _, _, _ = self.step(fire_action)
        state = observation # (batch, state_size)
        return state
        
    def step(self, action): 
        # action is [0, 1, 2] 0: NOOP, 1: RIGHT -> 2, 2: LEFT -> 3
        action = self.action_list[action]
        observation, reward, done, info = self.env.step(action)
        state = observation
        #print(observation.squeeze(0).shape)
        #plt.imshow(observation.squeeze(0).squeeze(0))
        #plt.show()

        if done:
            state = None

        return state, reward, done, info
