from abc import *
from typing import NamedTuple, List
from collections import namedtuple
import numpy as np
from lib.sumtree import SumTree
import ray

#################################
#####     Replay Memory    ######
#################################

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
                        
class IReplay(metaclass=ABCMeta):
    @abstractmethod
    def push(self, transition: Transition):
        '''データの挿入'''
        pass
    @abstractmethod
    def sample(self, batch_size):
        '''データの抽出'''
        pass
    @abstractmethod
    def update_priority(self, state_action_values, expected_state_action_values):
        '''なにかしらの処理'''
        pass
    @abstractmethod
    def refresh(self):
        '''なにかしらの処理'''
        pass

class ReplayParameter(NamedTuple):
    capacity: int
    epsilon: float
    alpha: float

@ray.remote
class Replay(IReplay):

    def __init__(self, param: ReplayParameter):
        self.capacity = param.capacity  # メモリの最大長さ
        self.tree = SumTree(param.capacity)
        self.epsilon = param.epsilon
        self.alpha = param.alpha
        self.cycle = 0
        self.cycle_size = 0
        self.size = 0

    def _getPriority(self, td_error):
        return (td_error + self.epsilon) ** self.alpha
 
    def push(self, td_errors ,transitions: List[Transition]):
        assert len(td_errors) == len(transitions)
        #print("replay size2: ", self.size)
        priorities = (np.abs(td_errors) + self.epsilon) ** self.alpha
        #print(td_errors.shape)
        for priority, transition in zip(priorities, transitions):
            """state, action, state_next, rewardをメモリに保存します"""
            self.size += 1
            if self.size > self.capacity:
                self.size = self.capacity

            
            self.cycle_size += 1
            if self.cycle_size % self.capacity == self.capacity-1:
                self.cycle_size = 0
                self.cycle += 1
                print("Replay cycle: {}".format(self.cycle))
            #print(priority.shape, transition)
            self.tree.add(priority, transition)
 
    def sample(self, batch_size):
        #print("replay size: ", self.size)
        transitions = []
        sampled_indices = []

        for rand in np.random.uniform(0, self.tree.total(), batch_size):
            (idx, _, data) = self.tree.get(rand)
            transitions.append(data)
            sampled_indices.append(idx)

        return sampled_indices, transitions

    def update_priority(self, indices, td_errors):
        assert len(indices) == len(td_errors)
        #print(td_errors)
        priorities = (np.abs(td_errors) + self.epsilon) ** self.alpha
        for idx, priority in zip(indices, priorities):
            self.tree.update(idx, priority)

    
    def refresh(self):
        self.tree = SumTree(self.capacity)
        self.size = 0
 
    def length(self):
        return self.size
