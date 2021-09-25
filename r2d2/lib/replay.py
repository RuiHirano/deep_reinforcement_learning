import sys
sys.path.append('./../')
from abc import *
from typing import NamedTuple, List
from collections import namedtuple
import numpy as np
from lib.sumtree import SumTree
from lib.episode_buffer import Segment

#################################
#####     Replay Memory    ######
#################################
                        
class IReplay(metaclass=ABCMeta):
    @abstractmethod
    def push(self):
        '''データの挿入'''
        pass
    @abstractmethod
    def sample(self):
        '''データの抽出'''
        pass
    @abstractmethod
    def update_priority(self):
        '''なにかしらの処理'''
        pass

class ReplayParameter(NamedTuple):
    capacity: int

class SegmentReplay(IReplay):

    def __init__(self, param: ReplayParameter):
        self.capacity = param.capacity  # メモリの最大長さ
        self.tree = SumTree(param.capacity)
        self.cycle = 0
        self.cycle_size = 0
        self.size = 0
        self.indexes = []

    def push(self, priorities, segments: List[Segment]):
        assert len(priorities) == len(segments)
        for priority, segment in zip(priorities, segments):
            """state, action, state_next, rewardをメモリに保存します"""
            self.size += 1
            if self.size > self.capacity:
                self.size = self.capacity

            self.cycle_size += 1
            if self.cycle_size % self.capacity == self.capacity-1:
                self.cycle_size = 0
                self.cycle += 1
                print("Replay cycle: {}".format(self.cycle))
                
            self.tree.add(priority, segment)
 
    def sample(self, batch_size):
        """batch_size分だけ、ランダムに保存内容を取り出します"""
        segments = []
        sampled_indices = []
        for rand in np.random.uniform(0, self.tree.total(), batch_size):
            (idx, _, data) = self.tree.get(rand)
            segments.append(data)
            sampled_indices.append(idx)

        return sampled_indices, segments

    def update_priority(self, indices, priorities):
        assert len(indices) == len(priorities)
        for idx, priority in zip(indices, priorities):
            self.tree.update(idx, priority)
 
    def __len__(self):
        return self.size

if __name__ == "__main__":
    pass
    '''replay = SegmentReplay(ReplayParameter(
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
        print("{}: {}".format(td_errors[key], keys[key]*100/10000))'''
    