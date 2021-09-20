from abc import *
from typing import NamedTuple
import torch
import random
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
import os
import copy
from .replay import Transition, IReplay

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#################################
#####     Brain Interface  ######
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
#####        Brain         ######
#################################

class BrainParameter(NamedTuple):
    batch_size: int
    gamma : float
    eps_start : float
    eps_end: float
    eps_decay: int
    replay: IReplay
    net: nn.Module
    multi_step_bootstrap: bool = False
    num_multi_step_bootstrap: int = 5

class Brain(IBrain):
    def __init__(self, param, num_actions):
        self.steps_done = 0

        # muti-step bootstrap
        print("bootstrap: {}".format(param.multi_step_bootstrap))
        self.multi_step_bootstrap = param.multi_step_bootstrap
        self.num_multi_step_bootstrap = param.num_multi_step_bootstrap
        self.multi_step_transitions = []
        
        # Brain Parameter
        self.BATCH_SIZE = param.batch_size
        self.GAMMA = param.gamma
        self.EPS_START = param.eps_start
        self.EPS_END = param.eps_end
        self.EPS_DECAY = param.eps_decay
        
        # 経験を保存するメモリオブジェクトを生成
        self.replay = param.replay
        
        #print(self.model) # ネットワークの形を出力
        self.num_actions = num_actions
        #print(self.num_observ)
        self.policy_net = copy.deepcopy(param.net).to(device)
        self.target_net = copy.deepcopy(param.net).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        # 最適化手法の設定
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0001)
        
    def optimize(self):
        if len(self.replay) < self.BATCH_SIZE:
            return
        
        ''' batch化する '''
        transitions = self.replay.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state).to(device) # state: tensor([[0.5, 0.4, 0.5, 0], ...]) size(32, 4)
        action_batch = torch.cat(batch.action).to(device) # action: tensor([[1],[0],[0]...]) size(32, 1) 
        reward_batch = torch.cat(batch.reward).to(device) # reward: tensor([1, 1, 1, 0, ...]) size(32)
        #next_state_batch = torch.cat(batch.next_state) # next_state: tensor([[0.5, 0.4, 0.5, 0], ...]) size(32, 4)
        #assert state_batch.size() == (self.BATCH_SIZE,self.num_states) # TODO: fix assertion
        #assert next_state_batch.size() == (self.BATCH_SIZE,self.num_states)
        assert action_batch.size() == (self.BATCH_SIZE,1)
        assert reward_batch.size() == (self.BATCH_SIZE,)


        ''' 出力データ：行動価値を作成 '''
        # 出力actionの値のうちaction_batchが選んだ方を抽出（.gather()）
        state_action_values = self.policy_net(state_batch).gather(1, action_batch) # size(32, 1)
        #print(state_action_values.size())
        assert state_action_values.size() == (self.BATCH_SIZE,1)

        ''' 教師データを作成する '''
        ''' target = 次のステップでの行動価値の最大値 * 時間割引率 + 即時報酬 '''
         # doneされたかどうか doneであればfalse
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=device, dtype=torch.bool) # non_final_mask: tensor([True, True, True, False, ...]) size(32)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None]).to(device) # size(32-None num,4)
        assert non_final_mask.size() == (self.BATCH_SIZE,)
        #print(non_final_next_states.size())
        #assert non_final_next_states.size() == (self.BATCH_SIZE,self.num_states)
        

        # 次の環境での行動価値
        next_state_values = torch.zeros(self.BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach() # size(32)
        assert next_state_values.size() == (self.BATCH_SIZE,)

        # target = 次のステップでの行動価値の最大値 * 時間割引率 + 即時報酬
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch
        if self.multi_step_bootstrap:
            expected_state_action_values = (next_state_values * (self.GAMMA ** self.num_multi_step_bootstrap)) + reward_batch
        assert expected_state_action_values.size() == (self.BATCH_SIZE,)

        ''' Loss を計算'''
        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        #print(expected_state_action_values.unsqueeze(1))
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        ''' 勾配計算、更新 '''
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        ''' memoryのupdate処理(PERではpriorityを更新) '''
        td_errors = (expected_state_action_values.unsqueeze(1) - state_action_values).squeeze(1).detach().cpu().numpy()
        self.replay.update(td_errors)

        return loss.item()
    
    def memorize(self, transition):
        '''memoryオブジェクトに、state, action, state_next, rewardの内容を保存する'''
        # multi step bootstrap
        if self.multi_step_bootstrap:
            transition = self._get_multi_step_transition(transition)
            if transition is not None:
                self.replay.push(transition)
        else:
            self.replay.push(transition)

    def _get_multi_step_transition(self, transition):
        # 計算用のバッファに遷移を登録
        self.multi_step_transitions.append(transition)
        if len(self.multi_step_transitions) < self.num_multi_step_bootstrap:
            return None

        next_state = transition.next_state
        nstep_reward = 0
        for i in range(self.num_multi_step_bootstrap):
            r = self.multi_step_transitions[i].reward
            nstep_reward += r * self.GAMMA ** i

            # 終端の場合、それ以降の遷移は次のepisodeのものなので計算しない
            if self.multi_step_transitions[i].next_state is None:
                next_state = None
                break

        # 最も古い遷移を捨てる
        state, action, _, _ = self.multi_step_transitions.pop(0)
    
        # 時刻tでのstateとaction、t+nでのstate、その間での報酬の和をreplay memoryに登録
        return Transition(state, action, next_state, nstep_reward)


    def update_target_model(self):
        # モデルの重みをtarget_networkにコピー
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def decide_action(self, state):
        state = torch.tensor(state, device=device).float()
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                action = np.argmax(self.policy_net(state).tolist()) 
        else:
            # 0,1の行動をランダムに返す
            action = random.randrange(self.num_actions)  
        return action
    
    
    def save_model(self, name):
        dirname = os.path.dirname(name)

        print("name", name)
        print("dirname", dirname)
        if os.path.exists(dirname) == False:
            os.makedirs(dirname)
        torch.save(self.policy_net.state_dict(), name)
        
    def read_model(self, name):
        param = torch.load(name)
        self.policy_net.load_state_dict(param)
        self.target_net.load_state_dict(param)
    
    def predict(self, state):
        state = torch.tensor(state, device=device).float()
        self.target_net.eval() # ネットワークを推論モードに切り替える
        with torch.no_grad():
            action = np.argmax(self.target_net(state).tolist())
        return action

