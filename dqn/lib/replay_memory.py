from abc import *
from typing import NamedTuple
import torch
from collections import namedtuple
import random
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
import os
import copy
from itertools import count
import time
import matplotlib.pyplot as plt
from .interface import IReplayMemory, Transition

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#################################
#####     Replay Memory    ######
#################################

class ReplayMemory(IReplayMemory):
 
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

class PrioritizedReplayMemory(IReplayMemory):

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

class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros( 2*capacity - 1 )
        self.data = np.zeros( capacity, dtype=object )
        self.index_leaf_start = capacity - 1

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def max(self):
        return self.tree[self.index_leaf_start:].max()

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])