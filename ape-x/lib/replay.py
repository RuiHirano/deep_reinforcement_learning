from abc import *
from typing import NamedTuple, List
from collections import namedtuple
import numpy as np
from .sumtree import SumTree

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

class ReplayParameter(NamedTuple):
    capacity: int
    epsilon: float
    alpha: float

class Replay(IReplay):

    def __init__(self, param: ReplayParameter):
        self.capacity = param.capacity  # メモリの最大長さ
        self.tree = SumTree(param.capacity)
        self.epsilon = param.epsilon
        self.alpha = param.alpha
        self.cycle = 0
        self.cycle_size = 0
        self.size = 0
        self.indexes = []

    def _getPriorities(self, td_errors):
        return (np.abs(td_errors) + self.epsilon) ** self.alpha
 
    def push(self, td_errors ,transitions: List[Transition]):
        assert len(td_errors) == len(transitions)
        priorities = self._getPriorities(td_errors)
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
        """batch_size分だけ、ランダムに保存内容を取り出します"""
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
        priorities = self._getPriorities(td_errors)
        for idx, priority in zip(indices, priorities):
            self.tree.update(idx, priority)
 
    def __len__(self):
        return self.size

if __name__ == "__main__":
    replay = Replay(ReplayParameter(
        capacity=10000,
        epsilon=0.001,
        alpha=0.6
    ))
    td_errors = [5,10,15, 20]
    transitions = [Transition(i, i, i, i) for i in range(4)]
    replay.push(td_errors, transitions)
    
    keys = {}
    for i in range(10000):
        indices, transitions = replay.sample(1)
        if transitions[0].state not in keys:
            keys[transitions[0].state] = 0
        else:
            keys[transitions[0].state] += 1

    for key in keys.keys():
        print("{}: {}".format(td_errors[key], keys[key]*100/10000))

    # update priority
    indices, transitions = replay.sample(1)
    print("update {} index priority {} to 50".format(transitions[0].state, td_errors[transitions[0].state]))
    new_td_errors = [50]
    replay.update_priority(indices, new_td_errors)

    keys = {}
    for i in range(10000):
        indices, transitions = replay.sample(1)
        if transitions[0].state not in keys:
            keys[transitions[0].state] = 0
        else:
            keys[transitions[0].state] += 1

    for key in keys.keys():
        print("{}: {}".format(td_errors[key], keys[key]*100/10000))
    