from abc import *
import time
from typing import NamedTuple
import torch
import torch.nn as nn
import copy
from .replay import Transition
import ray
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
    gamma : float
    net: nn.Module
    optimizer: torch.optim.Optimizer
    num_multi_step_bootstrap: int  # multi-step bootstrapのステップ数

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

    def update_network(self, minibatchs):
        #time.sleep(0.05)
        indices_all = []
        td_errors_all = []

        for (indices, transitions) in minibatchs:
            #print(indices)

            batch = Transition(*zip(*transitions))
            state_batch = torch.cat(batch.state).to(device) # state: tensor([[0.5, 0.4, 0.5, 0], ...]) size(32, 4)
            action_batch = torch.Tensor(list(batch.action)).unsqueeze(1).type(torch.int64).to(device) # action: tensor([[1],[0],[0]...]) size(32, 1) 
            reward_batch = torch.Tensor(list(batch.reward)).to(device) # reward: tensor([1, 1, 1, 0, ...]) size(32)
            done_batch = torch.Tensor(list(batch.done)).to(device) # reward: tensor([T, F, F, T, ...]) size(32)

            ''' 出力データ：行動価値を作成 '''
            Q = self.policy_net(state_batch).gather(1, action_batch) # size(32, 1)
            assert Q.size() == (self.batch_size,1)

            ''' 教師データを作成する '''
            ''' target = 次のステップでの行動価値の最大値 * 時間割引率 + 即時報酬 '''
            non_final_mask = torch.logical_not(done_batch) # non_final_mask: tensor([True, True, True, False, ...]) size(32)
            non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(device) # size(32-None num,4)
            next_state_values = torch.zeros(self.batch_size, device=device)
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach() # size(32)
            TQ = (next_state_values * (self.gamma ** self.num_multi_step_bootstrap)) + reward_batch
            assert TQ.size() == (self.batch_size,)

            '''td errorsを求める'''
            td_errors = (TQ.unsqueeze(1) - Q).squeeze(1).detach().cpu().numpy()
            assert td_errors.shape == (self.batch_size, )

            ''' Loss を計算'''
            # Compute Huber loss
            criterion = nn.SmoothL1Loss()
            loss = criterion(Q, TQ.unsqueeze(1))

            ''' 勾配計算、更新 '''
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            indices_all.extend(indices)
            td_errors_all.extend(td_errors)
            
        assert len(td_errors_all) == self.batch_size*len(minibatchs)
        assert len(indices_all) == self.batch_size*len(minibatchs)
        current_weights = self.policy_net.state_dict()
        self.target_net.load_state_dict(current_weights)
        
        # actor, testerのためにweightsをcpuに変更する
        current_weights = copy.deepcopy(self.policy_net).to("cpu").state_dict()
        
        return current_weights, indices_all, td_errors_all, loss.item()

    def get_current_weights(self):
        # actor, testerのためにweightsをcpuに変更する
        current_weights = copy.deepcopy(self.target_net).to("cpu").state_dict()
        return current_weights

    def save_model(self, name):
        dirname = os.path.dirname(name)
        if os.path.exists(dirname) == False:
            os.makedirs(dirname)
        torch.save(self.target_net.state_dict(), name)

