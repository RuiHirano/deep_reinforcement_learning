from abc import *
from collections import namedtuple
import random
import numpy as np
from .sumtree import SumTree

###########################################
#####     Replay Memory Interface    ######
###########################################

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

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
    def update(self, td_errors):
        '''なにかしらの処理'''
        pass
    

##################################
#####     Replay Memory     ######
##################################

class ReplayMemory(IReplay):
 
    def __init__(self, CAPACITY):
        self.capacity = CAPACITY  # メモリの最大長さ
        self.memory = []  # 経験を保存する変数
        self.index = 0  # 保存するindexを示す変数
 
    def push(self, transition: Transition):
        """state, action, state_next, rewardをメモリに保存します"""
 
        if len(self.memory) < self.capacity:
            self.memory.append(None)  # メモリが満タンでないときは足す
 
        # namedtupleのTransitionを使用し、値とフィールド名をペアにして保存します
        self.memory[self.index] = transition
        self.index = (self.index + 1) % self.capacity  # 保存するindexを1つずらす
 
    def sample(self, batch_size):
        """batch_size分だけ、ランダムに保存内容を取り出します"""
        return random.sample(self.memory, batch_size)

    def update(self, td_errors):
        pass
 
    def __len__(self):
        return len(self.memory)


###########################################
#####     Prioritized Replay Memory  ######
###########################################

class PrioritizedReplayMemory(IReplay):

    def __init__(self, CAPACITY, epsilon = 0.0001, alpha = 0.6):
        self.capacity = CAPACITY  # メモリの最大長さ
        self.tree = SumTree(CAPACITY)
        self.epsilon = epsilon
        self.alpha = alpha
        self.size = 0
        self.indexes = []

    def _getPriority(self, td_error):
        return (np.abs(td_error) + self.epsilon) ** self.alpha
 
    def push(self, transition: Transition):
        """state, action, state_next, rewardをメモリに保存します"""
        self.size += 1
        if self.size > self.capacity:
            self.size = self.capacity

        priority = self.tree.max()
        if priority <= 0:
            priority = 1

        self.tree.add(priority, transition)
 
    def sample(self, batch_size):
        """batch_size分だけ、ランダムに保存内容を取り出します"""
        list = []
        self.indexes = []
        for rand in np.random.uniform(0, self.tree.total(), batch_size):
            (idx, _, data) = self.tree.get(rand)
            list.append(data)
            self.indexes.append(idx)

        return list

    def update(self, td_errors):
        if self.indexes != None:
            for i, td_error in enumerate(td_errors):
                priority = self._getPriority(td_error)
                self.tree.update(self.indexes[i], priority)

    def update_priority(self, idx, td_error):
        priority = self._getPriority(td_error)
        self.tree.update(idx, priority)
 
    def __len__(self):
        return self.size
