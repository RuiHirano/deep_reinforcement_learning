import sys
sys.path.append('./../')
from abc import *
from typing import NamedTuple, List
from numpy.core.fromnumeric import mean
import torch
import random
import torch.nn as nn
import numpy as np
import ray
import gym
import copy
import time

# if gpu is to be used
device = "cpu"

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
    render: bool


@ray.remote(num_cpus=1)
class Tester(ITester):
    def __init__(self, param: TesterParameter):
        
        self.env = param.env
        self.action_space = self.env.action_space.n
        self.q_network = copy.deepcopy(param.net).to(device)
        self.q_network.eval()
        self.num_test_episode = param.num_test_episode
        self.render = param.render

    def test_play(self, current_weights):
        #print("tester", current_weights['fcA1.weight'][0][0])
        self.q_network.load_state_dict(current_weights)

        episode_rewards_all = []
        state = self.env.reset()

        for i in range(self.num_test_episode):
            state = self.env.reset()
            episode_rewards = 0
            done = False
            prev_action = 0
            while not done:
                if self.render:
                    self.env.render()
                x, _ = self.q_network(state, (None, None), prev_action)
                action = np.argmax(x.tolist())  # TODO
                next_state, reward, done, _ = self.env.step(action)
                episode_rewards += reward
                state = next_state
            episode_rewards_all.append(episode_rewards)

        mean_test_score = np.array(episode_rewards_all).mean()
        print(episode_rewards_all)
        return mean_test_score
