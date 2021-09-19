from abc import *
from .brain import IBrain

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
#####        Agent         ######
#################################

class Agent(IAgent):
    def __init__(self, brain: IBrain):
        '''エージェントが行動を決定するための頭脳を生成'''
        self.brain = brain
        
    def learn(self):
        '''Q関数を更新する'''
        loss = self.brain.optimize()
        return loss
        
    def modify_target(self):
        '''Target Networkを更新する'''
        self.brain.update_target_model()
        
    def select_action(self, state):
        '''行動を決定する'''
        action = self.brain.decide_action(state)
        return action
    
    def memorize(self, transition):
        '''memoryオブジェクトに、state, action, state_next, rewardの内容を保存する'''
        self.brain.memorize(transition)
    
    def predict_action(self, state):
        '''行動を予測する'''
        action = self.brain.predict(state)
        return action
    
    def record(self, name):
        '''モデルを保存する'''
        self.brain.save_model(name)
        
    def remember(self, name):
        '''モデルを読み込む'''
        self.brain.read_model(name)
