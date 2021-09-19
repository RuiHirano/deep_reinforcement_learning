from abc import *
import time
from typing import NamedTuple
import torch
import random
import torch.nn as nn
import numpy as np
from .replay import Transition, IReplay
import ray
import gym
import copy

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#################################
#####        Weights         ######
#################################

class IWeights(metaclass=ABCMeta):
    @abstractmethod
    def set(self):
        pass
    def get(self):
        pass
    def is_update(self, pid: int):
        pass

@ray.remote
class Weights(IWeights):
    def __init__(self, init_weights):
        self.weights = init_weights
        self.pids = []

    def get(self):
        return self.weights

    def set(self, weights):
        self.weights = weights
        self.pids = []

    def is_update(self, pid: int):
        if pid not in self.pids:
            self.pids.append(pid)
            return True
        return False