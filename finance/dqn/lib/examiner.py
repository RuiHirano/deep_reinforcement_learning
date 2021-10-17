from abc import *
from itertools import count
import time
import matplotlib.pyplot as plt
import torch
import copy

#################################
#####  Examiner Interface  ######
#################################

class IExaminer(metaclass=ABCMeta):
    @abstractmethod
    def evaluate(self):
        '''評価'''
        pass


#################################
#####        Examiner      ######
#################################

class Examiner():
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        
    def change_state(self, state):
        '''change state Tensor(L, N, Hin)'''
        state = torch.tensor(state.values).float() # (10, 5)
        # extract Close data
        state = state[:, 3].unsqueeze(1) # OHLCV 3 is Close (10, 1)
        state = state.unsqueeze(1) # add batch (10, 1, 1) (L, B, Hin)
        return state

    def evaluate(self, episode_num, render=True):
        state = self.env.reset()
        state = self.change_state(state)
        #state = torch.from_numpy(state).type(torch.FloatTensor)  # numpy変数をPyTorchのテンソルに変換
        #state = torch.unsqueeze(state, 0)

        start = time.time()
        done = False
        step = 0
        while not done:
            step += 1
            print(step)
            ''' 行動を決定する '''
            action = self.agent.predict_action(state)
            ''' 行動に対する環境や報酬を取得する '''
            next_state, _, done, _ = self.env.step(action)  # state [0,0,0,0...window_size], reward 1.0, done False, input: action 0 or 1 or 2
            next_state = self.change_state(next_state)
            if step > 2000:
                break

            state = next_state

        elapsed_time = time.time() - start
        ''' 終了時に結果をプロット '''
        #print(loss, episode_i)
        print("Time: {} [sec]".format("{:.2f}".format(elapsed_time)))

            
        ''' 結果を出力する '''
        print('Completed!')
        stats = self.env.stats()
        print(stats)
        self.env.plot()