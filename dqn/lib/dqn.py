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
import datetime
from .replay_memory import Transition
from .interface import IAgent, IBrain, IExaminer, IReplayMemory, ITrainer
from torch.utils.tensorboard import SummaryWriter

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#################################
#####        Brain         ######
#################################

class BrainParameter(NamedTuple):
    batch_size: int
    gamma : float
    eps_start : float
    eps_end: float
    eps_decay: int
    replay_memory: IReplayMemory
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
        self.memory = param.replay_memory
        
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
        if len(self.memory) < self.BATCH_SIZE:
            return
        
        ''' batch化する '''
        transitions = self.memory.sample(self.BATCH_SIZE)
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
        td_errors = (expected_state_action_values.unsqueeze(1) - state_action_values).squeeze(1).detach().numpy()
        self.memory.update(td_errors)

        return loss.item()
    
    def memorize(self, transition):
        '''memoryオブジェクトに、state, action, state_next, rewardの内容を保存する'''
        # multi step bootstrap
        if self.multi_step_bootstrap:
            transition = self._get_multi_step_transition(transition)
            if transition is not None:
                self.memory.push(transition)
        else:
            self.memory.push(transition)

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

#################################
#####        Trainer         ######
#################################


class TrainParameter(NamedTuple):
    target_update_iter: int
    num_episode : int
    save_iter: int
    save_filename: str
    render: bool


class Trainer(ITrainer):
    def __init__(self, id, env, agent):
        self.id = id
        self.env = env
        self.agent = agent
        self.loss_durations = []
        self.episode_durations = []
        self.log_dir = "results/{}".format(self.id)
        self.writer = SummaryWriter(self.log_dir)
        
    def train(self, train_param):
        train_start = time.time()
        for episode_i in range(train_param.num_episode):
            state = self.env.reset()
            #state = torch.from_numpy(state).type(torch.FloatTensor)  # numpy変数をPyTorchのテンソルに変換
            #state = torch.unsqueeze(state, 0)

            start = time.time()
            sum_loss = 0
            for step in count():
                if train_param.render:
                    self.env.render()

                ''' 行動を決定する '''
                action = self.agent.select_action(state) # input ex: <list> [0, 0, 0, 0], output ex: <int> 0 or 1
                #print("action", action)
                ''' 行動に対する環境や報酬を取得する '''
                next_state, reward, done, _ = self.env.step(action)  # state [0,0,0,0...window_size], reward 1.0, done False, input: action 0 or 1 or 2

                ''' エージェントに記憶させる '''
                self.agent.memorize(
                    Transition(
                        state, 
                        torch.tensor([[action]], device=device), 
                        next_state, 
                        torch.tensor([reward], device=device)
                    )
                )

                # Move to the next state
                state = next_state

                ''' エージェントに学習させる '''
                loss = self.agent.learn()
                if loss != None:
                    sum_loss += loss

                if done:
                    elapsed_time = time.time() - start
                    ''' 終了時に結果をプロット '''
                    #print(loss, episode_i)
                    print("Episode: {}, Step: {}, Loss: {}, Time: {} [sec]".format(episode_i, step, sum_loss/step, "{:.2f}".format(elapsed_time)))
                    #self.episode_durations.append(step + 1)
                    self.writer.add_scalar('reward', step+1, episode_i)
                    self.writer.add_scalar('training loss',sum_loss/step, episode_i)

                    
                    # 次のエピソードへ
                    break
            # Update the target network, copying all weights and biases in DQN
            if episode_i % train_param.target_update_iter == 0:
                ''' 目標を修正する '''
                print("Episode: {}, Update Target Net".format(episode_i))
                self.agent.modify_target()

            if episode_i % train_param.save_iter == train_param.save_iter-1:
                fn = "./results/{}/{}_{}.pth".format(self.id, train_param.save_filename, episode_i+1)
                self.agent.record(fn)
                print('Saved model! Name: {}'.format(fn))

        ''' モデルを保存する '''
        # モデルの保存
        fn = "./results/{}/{}_{}.pth".format(self.id, train_param.save_filename, episode_i+1)
        self.agent.record(fn)
        print('Saved model! Name: {}'.format(fn))
        elapsed_time = time.time() - train_start
        print('Completed!, Time: {}'.format(elapsed_time))
        #self.plot_durations()
        
        
    def plot_durations(self):
        fig = plt.figure()
        ax = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)
        x = [s for s in range(len(self.loss_durations))]
        y = self.loss_durations
        x2 = [s for s in range(len(self.episode_durations))]
        y2 = self.episode_durations
        ax.plot(x, y, color="red", label="loss")
        ax2.plot(x2, y2, color="blue", label="episode")
        ax.legend(loc = 'upper right') #凡例
        ax2.legend(loc = 'upper right') #凡例
        fig.tight_layout()              #レイアウトの設定
        plt.show()


#################################
#####        Examiner      ######
#################################

class Examiner():
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.episode_durations = []
        
    def evaluate(self, episode_num, render=True):
        for episode_i in range(episode_num):
            state = self.env.reset()
            #state = torch.from_numpy(state).type(torch.FloatTensor)  # numpy変数をPyTorchのテンソルに変換
            #state = torch.unsqueeze(state, 0)

            start = time.time()
            for step in count():
                if render:
                    self.env.render()

                ''' 行動を決定する '''
                action = self.agent.predict_action(state)
                ''' 行動に対する環境や報酬を取得する '''
                next_state, _, done, _ = self.env.step(action)  # state [0,0,0,0...window_size], reward 1.0, done False, input: action 0 or 1 or 2

                state = next_state

                ''' 終了時はnext_state_valueをNoneとする '''
                # Observe new state
                if done:
                    elapsed_time = time.time() - start
                    ''' 終了時に結果をプロット '''
                    #print(loss, episode_i)
                    print("Episode: {}, Step: {}, Time: {} [sec]".format(episode_i, step, "{:.2f}".format(elapsed_time)))
                    self.episode_durations.append(step + 1)
                    # 次のエピソードへ
                    break

            
        ''' 結果を出力する '''
        print('Completed!')
        self.plot_durations()

    def plot_durations(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        x = [s for s in range(len(self.episode_durations))]
        y = self.episode_durations
        ax.plot(x, y, color="blue", label="episode")
        ax.legend(loc = 'upper right') #凡例
        fig.tight_layout()              #レイアウトの設定
        plt.show()