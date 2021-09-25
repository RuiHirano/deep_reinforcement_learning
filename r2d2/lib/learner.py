import sys
sys.path.append('./../')
from abc import *
import time
from typing import NamedTuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from lib.episode_buffer import EpisodeBuffer, Transition, Segment
import ray
import gym
import os
# https://horomary.hatenablog.com/entry/2021/03/02/235512#Learner%E3%81%AE%E5%BD%B9%E5%89%B2
# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#################################
#####        Learner         ######
#################################

class ILearner(metaclass=ABCMeta):
    @abstractmethod
    def get_current_weights(self):
        '''Q関数の最適化'''
        pass
    @abstractmethod
    def update_network(self):
        '''Target Networkの更新'''
        pass
    @abstractmethod
    def save_model(self):
        pass

class LearnerParameter(NamedTuple):
    batch_size: int
    env : gym.Wrapper
    gamma : float
    net: nn.Module
    optimizer: torch.optim.Optimizer
    num_multi_step_bootstrap: int  # multi-step bootstrapのステップ数
    burnin_len: int
    unroll_len: int
    eta: float
    alpha: float
    priority_epsilon: float

@ray.remote(num_cpus=1, num_gpus=1 if torch.cuda.is_available() else 0)
class Learner(ILearner):
    def __init__(self, param):
        self.batch_size = param.batch_size
        self.gamma = param.gamma
        self.policy_net = param.net.to(device)
        self.target_net = copy.deepcopy(param.net).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = param.optimizer
        self.num_multi_step_bootstrap = param.num_multi_step_bootstrap
        self.burnin_len = param.burnin_len
        self.unroll_len = param.unroll_len
        self.eta = param.eta
        self.alpha = param.alpha
        self.env = param.env
        self.action_space = self.env.action_space.n
        self.num_states = self.env.observation_space.shape[0]
        self.priority_epsilon = param.priority_epsilon

    def update_network(self, minibatchs):
        #time.sleep(0.05)
        indices_all = []
        priorities_all = []
        losses = []

        for (indices, segments) in minibatchs:
            #print(indices)
            batch = Segment(*zip(*segments))
            batch_size = len(segments)
            states_batch = torch.stack(batch.states).permute(1,0,2).to(device) # (timestep, batch_size state_size)
            actions_batch = torch.stack(batch.actions).permute(1,0,2).type(torch.int64).to(device) # (timestep, batch_size, 1) 
            rewards_batch = torch.stack(batch.rewards).permute(1,0).type(torch.float64).to(device) # (batch_size, unroll_len, 1)
            dones_batch = torch.stack(batch.rewards).permute(1,0).to(device) # (unroll_len, batch_size, 1) 全て0？
            assert states_batch.size() == (self.burnin_len+self.unroll_len, batch_size, self.num_states)
            assert actions_batch.size() == (self.burnin_len+self.unroll_len, batch_size,1)
            assert rewards_batch.size() == (self.unroll_len, batch_size)
            assert dones_batch.size() == (self.unroll_len, batch_size)

            #: Stored state
            c0_batch = torch.cat(batch.c_init, dim=1).to(device)  # (1, batch_size, lstm_out_dim)
            h0_batch = torch.cat(batch.h_init, dim=1).to(device)  # (1, batch_size, lstm_out_dim)
            a0_batch = torch.stack(batch.a_init).permute(1,0,2).to(device)  # (timestep[1], bacth_size, 1)
            prev_actions_batch = torch.cat([a0_batch, actions_batch], dim=0)[:-1]  # (timestep, batch_size, 1)
            last_states_batch = torch.stack(batch.last_state).permute(1,0,2).to(device) # (timestep[1], batch_size, state_size)
            next_states_batch = torch.cat([states_batch, last_states_batch], dim=0)[1:]  # (timestep, batch_size state_size)
            assert c0_batch.size() == (1, batch_size, c0_batch.size()[2])
            assert h0_batch.size() == (1, batch_size, h0_batch.size()[2])
            assert a0_batch.size() == (1, batch_size,1)
            assert prev_actions_batch.size() == (self.burnin_len+self.unroll_len, batch_size, 1)
            assert last_states_batch.size() == (1, batch_size, self.num_states)
            assert next_states_batch.size() == (self.burnin_len+self.unroll_len, batch_size, self.num_states)

            ''' 出力データ：行動価値を作成 '''
            #: burn-in
            c_batch, h_batch = c0_batch, h0_batch
            for t in range(self.burnin_len):
                _, (c_batch, h_batch) = self.policy_net(
                    states_batch[t], states=[c_batch, h_batch], prev_action=prev_actions_batch[t])

            # unroll
            qvalues = [] # [[0.2, 0.3, 0.4,0.3 ], [0.3, 0.2, ], ...] # (unroll_len, batch_size, action_space)
            for t in range(self.burnin_len, self.burnin_len+self.unroll_len):
                q, (c_batch, h_batch) = self.policy_net(
                    states_batch[t], states=[c_batch, h_batch], prev_action=prev_actions_batch[t])
                qvalues.append(q)
            qvalues = torch.stack(qvalues)   # (unroll_len, batch_size, action_space)

            # Q value
            actions_onehot = F.one_hot(actions_batch[self.burnin_len:], num_classes=self.action_space).squeeze(2) # (unroll_len, batch_size, action_space)
            Q = torch.sum(qvalues * actions_onehot, dim=(2), keepdims=False).to(torch.float64)  # (unroll_len, batch_size)
            assert Q.size() == (self.unroll_len, batch_size)

            ''' 教師データを作成する '''
            ''' target = 次のステップでの行動価値の最大値 * 時間割引率 + 即時報酬 '''
            #: burn-in
            c_batch, h_batch = c0_batch, h0_batch
            for t in range(self.burnin_len+1):
                _, (c_batch, h_batch) = self.target_net(
                    states_batch[t], states=[c_batch, h_batch], prev_action=prev_actions_batch[t])

            # unroll
            next_qvalues = []
            for t in range(self.burnin_len, self.burnin_len+self.unroll_len):
                q, (c_batch, h_batch) = self.target_net(
                    next_states_batch[t], states=[c_batch, h_batch], prev_action=actions_batch[t])
                next_qvalues.append(q)
            next_qvalues = torch.stack(next_qvalues)   # (unroll_len, batch_size, action_space)

            # Q value
            next_actions = torch.argmax(next_qvalues, dim=2)  # (unroll_len, batch_size)
            next_actions_onehot = F.one_hot(next_actions, num_classes=self.action_space).squeeze(2) # (unroll_len, batch_size, action_space)
            next_maxQ = torch.sum(next_qvalues * next_actions_onehot, dim=(2), keepdims=False)  # (unroll_len, batch_size)
            TQ = (rewards_batch + (self.gamma ** self.num_multi_step_bootstrap) * torch.logical_not(dones_batch) * next_maxQ).to(torch.float64)  # (unroll_len, batch_size)
            assert TQ.size() == (self.unroll_len, batch_size)

            '''td errorsを求める'''
            td_errors = torch.sub(TQ, Q).to(device).detach().numpy().copy()
            td_errors_abs = np.abs(td_errors)
            assert td_errors_abs.shape == (self.unroll_len, batch_size)

            priorities = self.eta * np.max(td_errors_abs, axis=0) \
                + (1 - self.eta) * np.mean(td_errors_abs, axis=0) # 最大値 * eta + (1-eta)*平均値
            priorities = (priorities + self.priority_epsilon) ** self.alpha
            assert priorities.shape == (batch_size,)

            ''' Loss を計算'''
            # Compute Huber loss
            criterion = nn.SmoothL1Loss()
            loss = criterion(Q, TQ)
            #print(loss.item())

            ''' 勾配計算、更新 '''
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())
            indices_all.extend(indices)
            priorities_all.extend(priorities)
            
        assert len(priorities_all) == len(indices_all)
        current_weights = self.policy_net.state_dict()
        self.target_net.load_state_dict(current_weights)
        loss_mean = np.array(losses).mean()
        #print(loss_mean)
        current_weights = copy.deepcopy(self.policy_net).to("cpu").state_dict()
        
        return current_weights, indices_all, priorities_all, loss_mean

    def get_current_weights(self):
        current_weights = copy.deepcopy(self.target_net).to("cpu").state_dict()
        return current_weights

    def save_model(self, name):
        dirname = os.path.dirname(name)
        if os.path.exists(dirname) == False:
            os.makedirs(dirname)
        torch.save(self.target_net.state_dict(), name)

