import sys
sys.path.append('./../')
from abc import *
from typing import NamedTuple
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lib.episode_buffer import EpisodeBuffer, Transition, Segment
import ray
import gym
import copy

# if gpu is to be used
device = "cpu"

#################################
#####        Actor         ######
#################################

class IActor(metaclass=ABCMeta):
    @abstractmethod
    def rollout(self):
        '''Q関数の更新'''
        pass

class ActorParameter(NamedTuple):
    pid: int
    env : gym.Wrapper
    net: nn.Module
    epsilon : float  # 探索率
    gamma: float    # 時間割引率
    num_multi_step_bootstrap: int  # multi-step bootstrapのステップ数
    burnin_len: int
    unroll_len: int
    eta: float
    alpha: float
    priority_epsilon: float


@ray.remote(num_cpus=1)
class Actor(IActor):
    def __init__(self, param: ActorParameter):
        
        self.pid = param.pid
        self.env = param.env
        self.action_space = self.env.action_space.n
        self.num_states = self.env.observation_space.shape[0]

        self.q_network = copy.deepcopy(param.net).to(device)
        self.q_network.eval()
        self.epsilon = param.epsilon
        self.priority_epsilon = param.priority_epsilon
        self.gamma = param.gamma
        
        self.num_multi_step_bootstrap = param.num_multi_step_bootstrap
        self.multi_step_transitions = []

        self.burnin_len = param.burnin_len
        self.unroll_len = param.unroll_len
        self.eta = param.eta
        self.alpha = param.alpha

    '''def select_action(self, state, c, h, prev_action):

        x, (c_n, h_n) = self.q_network(state, (c, h), prev_action)
        sample = random.random()
        if sample > self.epsilon:
            with torch.no_grad():
                action = np.argmax(x.tolist()) 
        else:
            # 0,1の行動をランダムに返す
            action = random.randrange(self.action_space)  
        return action, (c_n, h_n)'''

    def rollout(self, current_weights):
        ''' 1エピソードのrollout'''
        #: グローバルQ関数と重みを同期
        self.q_network.load_state_dict(current_weights)

        #: rollout 1 episode
        self.multi_step_transitions = []
        score = 0
        episode_buffer = EpisodeBuffer(burnin_length=self.burnin_len, unroll_length=self.unroll_len, gamma=self.gamma, num_multi_step_bootstrap=self.num_multi_step_bootstrap)
        state = self.env.reset()
        # c, h = self.q_network.lstm.get_initial_state(batch_size=1)
        h, c = None, None
        prev_action = 0
        done = False
        count=0
        while not done:
            count += 1
            #state = torch.unsqueeze(state, 0)
            action, (next_h, next_c) = self.q_network.select_action(state, (h, c), prev_action, self.epsilon)
            next_state, reward, done, _ = self.env.step(action)
            score += reward
            if h == None or c == None:
                transition = Transition(state, action, reward, next_state, done, next_h, next_c, prev_action)
            else:
                transition = Transition(state, action, reward, next_state, done, h, c, prev_action)
            episode_buffer.add(transition)

            state, h, c, prev_action = next_state, next_h, next_c, action

        '''Compute TD Error'''
        segments = episode_buffer.pull_segments()
        if len(segments) == 0:
            # segmentsが取得できない場合[]で返す
            return [], [], self.pid, score

        batch = Segment(*zip(*segments))
        batch_size = len(segments)
        states_batch = torch.stack(batch.states).permute(1,0,2).to(device) # (timestep, batch_size state_size)
        actions_batch = torch.stack(batch.actions).permute(1,0,2).type(torch.int64).to(device) # (timestep, batch_size, 1) 
        rewards_batch = torch.stack(batch.rewards).permute(1,0).type(torch.float64).to(device) # (batch_size, unroll_len, 1)
        dones_batch = torch.stack(batch.dones).permute(1,0).to(device) # (unroll_len, batch_size, 1) 全て0？
        assert states_batch.size() == (self.burnin_len+self.unroll_len, batch_size, self.num_states)
        assert actions_batch.size() == (self.burnin_len+self.unroll_len, batch_size,1)
        assert rewards_batch.size() == (self.unroll_len, batch_size)
        assert dones_batch.size() == (self.unroll_len, batch_size)

        #: Stored state
        c0_batch = torch.cat(batch.c_init, dim=1).to(device)  # (1, batch_size, lstm_out_dim)
        h0_batch = torch.cat(batch.h_init, dim=1).to(device)  # (1, batch_size, lstm_out_dim)
        a0_batch = torch.stack(batch.a_init).permute(1,0,2).to(device)  # (timestep[1], bacth_size, 1)
        prev_actions_batch = torch.cat([a0_batch, actions_batch], dim=0)[:-1]          # (timestep, batch_size, 1)
        last_states_batch = torch.stack(batch.last_state).permute(1,0,2).to(device) # (timestep[1], batch_size, state_size)
        assert c0_batch.size() == (1, batch_size, c0_batch.size()[2])
        assert h0_batch.size() == (1, batch_size, h0_batch.size()[2])
        assert a0_batch.size() == (1, batch_size,1)
        assert prev_actions_batch.size() == (self.burnin_len+self.unroll_len, batch_size, 1)
        assert last_states_batch.size() == (1, batch_size, self.num_states)

        #: burn-in
        h_batch, c_batch = h0_batch, c0_batch
        for t in range(self.burnin_len):
            _, (h_batch, c_batch) = self.q_network(
                states_batch[t], states=[h_batch, c_batch], prev_action=prev_actions_batch[t])

        # unroll
        qvalues = [] # [[0.2, 0.3, 0.4,0.3 ], [0.3, 0.2, ], ...] # (unroll_len, batch_size, action_space)
        for t in range(self.burnin_len, self.burnin_len+self.unroll_len):
            q, (h_batch, c_batch) = self.q_network(
                states_batch[t], states=[h_batch, c_batch], prev_action=prev_actions_batch[t])
            qvalues.append(q)
        qvalues = torch.stack(qvalues)   # (unroll_len, batch_size, action_space)

        # Q value
        actions_onehot = F.one_hot(actions_batch[self.burnin_len:], num_classes=self.action_space).squeeze(2) # (unroll_len, batch_size, action_space)
        Q = torch.sum(qvalues * actions_onehot, dim=(2), keepdims=False)  # (unroll_len, batch_size)
        assert Q.size() == (self.unroll_len, batch_size)

        # Target Q value
        remaining_qvalue, _ = self.q_network(last_states_batch[0], states=[h_batch, c_batch], prev_action=actions_batch[-1])
        remaining_qvalue = remaining_qvalue.unsqueeze(0)          # (1, batch_size, action_space)
        next_qvalues = torch.cat([qvalues[1:], remaining_qvalue], dim=0)    # (unroll_len, batch_size, action_space)
        next_actions = torch.argmax(next_qvalues, dim=2)  # (unroll_len, batch_size)
        next_actions_onehot = F.one_hot(next_actions, num_classes=self.action_space)    # (unroll_len, batch_size, action_space)
        next_maxQ = torch.sum(next_qvalues * next_actions_onehot, dim=2, keepdims=False)      # (unroll_len, batch_size)
        TQ = rewards_batch + (self.gamma ** self.num_multi_step_bootstrap) * torch.logical_not(dones_batch) * next_maxQ  # (unroll_len, batch_size)
        #print(dones_batch, torch.logical_not(dones_batch))
        #sys.exit(1)
        assert TQ.size() == (self.unroll_len, batch_size)

        td_errors = torch.sub(TQ, Q).to(device).detach().numpy().copy()
        td_errors_abs = np.abs(td_errors)
        assert td_errors_abs.shape == (self.unroll_len, batch_size)

        priorities = self.eta * np.max(td_errors_abs, axis=0) \
            + (1 - self.eta) * np.mean(td_errors_abs, axis=0) # 最大値 * eta + (1-eta)*平均値
        priorities = (priorities + self.priority_epsilon) ** self.alpha
        assert priorities.shape == (batch_size,)

        return priorities, segments, self.pid, score

    def _get_multi_step_transition(self, transition):
        # 計算用のバッファに遷移を登録
        self.multi_step_transitions.append(transition)
        if len(self.multi_step_transitions) < self.num_multi_step_bootstrap:
            return None

        nstep_reward = 0
        for i in range(self.num_multi_step_bootstrap):
            r = self.multi_step_transitions[i].reward
            nstep_reward += r * self.gamma ** i

            # 終端の場合、それ以降の遷移は次のepisodeのものなので計算しない
            if self.multi_step_transitions[i].next_state is None:
                break

        # 最も古い遷移を捨てる
        state, action, _, next_state, done, h, c, prev_action = self.multi_step_transitions.pop(0)
    
        # 時刻tでのstateとaction、t+nでのstate、その間での報酬の割引累積和をreplay memoryに登録
        return Transition(state, action, nstep_reward, next_state, done, h, c, prev_action)
