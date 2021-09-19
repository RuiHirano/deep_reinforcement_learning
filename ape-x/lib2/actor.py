from abc import *
import time
from typing import NamedTuple
import torch
import random
import torch.nn as nn
import numpy as np
from .replay import Transition, IReplay
import ray
import gym
import copy
from .weights import IWeights

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#################################
#####        Actor         ######
#################################

class IActor(metaclass=ABCMeta):
    @abstractmethod
    def run(self):
        '''Q関数の更新'''
        pass

class ActorParameter(NamedTuple):
    pid: int
    env : gym.Wrapper
    net: nn.Module
    epsilon : float  # 探索率
    gamma: float    # 時間割引率
    num_multi_step_bootstrap: int  # multi-step bootstrapのステップ数
    batch_size: int # batch数
    replay: IReplay
    weights: IWeights


@ray.remote
class Actor(IActor):
    def __init__(self, param: ActorParameter):
        
        self.pid = param.pid
        self.env = param.env
        self.action_space = self.env.action_space.n
        self.state = self.env.reset()
        self.weights = param.weights
        self.replay = param.replay

        self.q_network = copy.deepcopy(param.net).to(device)
        self.q_network.eval()
        self.epsilon = param.epsilon
        self.gamma = param.gamma
        
        self.num_multi_step_bootstrap = param.num_multi_step_bootstrap
        self.multi_step_transitions = []

        self.buffer = []
        self.batch_size = param.batch_size
        self.terminate_flag = False

    def select_action(self, state):
        sample = random.random()
        if sample > self.epsilon:
            with torch.no_grad():
                action = np.argmax(self.q_network(state).tolist()) 
        else:
            # 0,1の行動をランダムに返す
            action = random.randrange(self.action_space)  
        return action

    def run(self):
        print("start actor process")
        count = 0
        while True:
            #print(count)
            count += 1
            if ray.get(self.weights.is_update.remote(self.pid)):
                # weightsの更新
                weights = ray.get(self.weights.get.remote())
                self.q_network.load_state_dict(weights)
                self.buffer = []
                #print("update weights {}, {}".format(count, self.pid, weights['fcA1.weight'][0][0]))
                count = 0

            action = self.select_action(self.state)
            next_state, reward, done, _ = self.env.step(action)
            
            transition = Transition(
                        self.state, 
                        torch.tensor([[action]], device=device), 
                        next_state, 
                        torch.tensor([reward], device=device)
                    )
            transition = self._get_multi_step_transition(transition)
            if transition is not None:
                self.buffer.append(transition)

            # stateの更新
            if done: # TODO : doneとmultistepの挙動
                self.state = self.env.reset()
            else:
                self.state = next_state

            if len(self.buffer) >= self.batch_size:
                
                # Batch
                transitions = random.sample(self.buffer, self.batch_size)
                batch = Transition(*zip(*transitions))
                state_batch = torch.cat(batch.state).to(device) # state: tensor([[0.5, 0.4, 0.5, 0], ...]) size(32, 4)
                action_batch = torch.cat(batch.action).to(device) # action: tensor([[1],[0],[0]...]) size(32, 1) 
                reward_batch = torch.cat(batch.reward).to(device) # reward: tensor([1, 1, 1, 0, ...]) size(32)
                
                # Target Q value
                non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool) # non_final_mask: tensor([True, True, True, False, ...]) size(32)
                non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(device) # size(32-None num,4)
                next_state_values = torch.zeros(self.batch_size, device=device)
                next_state_values[non_final_mask] = self.q_network(non_final_next_states).max(1)[0].detach() # size(32)
                TQ = (next_state_values * (self.gamma ** self.num_multi_step_bootstrap)) + reward_batch

                # Q value
                Q = self.q_network(state_batch).gather(1, action_batch)
                td_errors = (TQ.unsqueeze(1) - Q).squeeze(1).detach().numpy()
                assert td_errors.shape == (self.batch_size,)
                ray.get(self.replay.push.remote(td_errors, transitions))
        

    def _get_multi_step_transition(self, transition):
        # 計算用のバッファに遷移を登録
        self.multi_step_transitions.append(transition)
        if len(self.multi_step_transitions) < self.num_multi_step_bootstrap:
            return None

        next_state = transition.next_state
        nstep_reward = 0
        for i in range(self.num_multi_step_bootstrap):
            r = self.multi_step_transitions[i].reward
            nstep_reward += r * self.gamma ** i

            # 終端の場合、それ以降の遷移は次のepisodeのものなので計算しない
            if self.multi_step_transitions[i].next_state is None:
                next_state = None
                break

        # 最も古い遷移を捨てる
        state, action, _, _ = self.multi_step_transitions.pop(0)
    
        # 時刻tでのstateとaction、t+nでのstate、その間での報酬の和をreplay memoryに登録
        return Transition(state, action, next_state, nstep_reward)

