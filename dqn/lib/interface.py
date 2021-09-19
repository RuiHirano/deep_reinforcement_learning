from abc import *
from collections import namedtuple

#################################
#####     Replay Memory    ######
#################################

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class IReplayMemory(metaclass=ABCMeta):
    @abstractmethod
    def push(self, transition: Transition):
        '''データの挿入'''
        pass
    @abstractmethod
    def sample(self, batch_size):
        '''データの抽出'''
        pass
    @abstractmethod
    def update(self, state_action_values, expected_state_action_values):
        '''なにかしらの処理'''
        pass

#################################
#####        Brain         ######
#################################

class IBrain(metaclass=ABCMeta):
    @abstractmethod
    def optimize(self):
        '''Q関数の最適化'''
        pass
    @abstractmethod
    def update_target_model(self):
        '''Target Networkの更新'''
        pass
    @abstractmethod
    def memorize(self, state, action, next_state, reward):
        '''ReplayMemoryへの保存'''
        pass
    @abstractmethod
    def decide_action(self):
        '''行動の決定'''
        pass
    @abstractmethod
    def save_model(self):
        '''modelの保存'''
        pass
    @abstractmethod
    def read_model(self):
        '''modelの保存'''
        pass
    @abstractmethod
    def predict(self):
        '''推論'''
        pass

#################################
#####        Agent         ######
#################################

class IAgent(metaclass=ABCMeta):
    @abstractmethod
    def learn(self):
        '''Q関数の更新'''
        pass
    @abstractmethod
    def modify_target(self):
        '''Target Networkの更新'''
        pass
    @abstractmethod
    def select_action(self):
        '''行動の決定'''
        pass
    @abstractmethod
    def memorize(self):
        '''memoryに、state, action, state_next, rewardの内容を保存'''
        pass
    @abstractmethod
    def predict_action(self):
        '''行動の予測'''
        pass
    @abstractmethod
    def record(self):
        '''モデルの保存'''
        pass
    @abstractmethod
    def remember(self):
        '''モデルの読み込み'''
        pass

#################################
#####        Trainer         ######
#################################

class ITrainer(metaclass=ABCMeta):
    @abstractmethod
    def train(self):
        '''訓練'''
        pass

#################################
#####        Examiner      ######
#################################

class IExaminer(metaclass=ABCMeta):
    @abstractmethod
    def evaluate(self):
        '''評価'''
        pass