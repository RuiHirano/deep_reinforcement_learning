from abc import *
from typing import NamedTuple, List
from numpy.core.fromnumeric import mean
import torch
import random
import torch.nn as nn
import numpy as np
from .replay import Transition
import ray
import gym
import copy

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#################################
#####        Tester         ######
#################################

class ITester(metaclass=ABCMeta):
    @abstractmethod
    def test_play(self):
        '''Q関数の更新'''
        pass

class TesterParameter(NamedTuple):
    env : gym.Wrapper
    net: nn.Module
    num_test_episode: int


@ray.remote
class Tester(ITester):
    def __init__(self, param: TesterParameter):
        
        self.env = param.env
        self.action_space = self.env.action_space.n
        self.q_network = copy.deepcopy(param.net).to(device)
        self.num_test_episode = param.num_test_episode

    def test_play(self, current_weights: List[float]):
        #print("tester", current_weights['fcA1.weight'][0][0])
        self.q_network.load_state_dict(current_weights)

        episode_rewards_all = []
        for i in range(self.num_test_episode):
            state = self.env.reset()
            episode_rewards = 0
            done = False
            while not done:
                action = np.argmax(self.q_network(state).tolist()) 
                next_state, reward, done, _ = self.env.step(action)
                episode_rewards += reward
                state = next_state
            episode_rewards_all.append(episode_rewards)

        mean_test_score = np.array(episode_rewards_all).mean()
        return mean_test_score
