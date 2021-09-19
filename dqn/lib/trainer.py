from abc import *
from typing import NamedTuple
import torch
from itertools import count
import time
from .replay import Transition
from torch.utils.tensorboard import SummaryWriter

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#################################
#####  Trainer Interface   ######
#################################

class ITrainer(metaclass=ABCMeta):
    @abstractmethod
    def train(self):
        '''訓練'''
        pass

#################################
#####        Trainer       ######
#################################

class TrainParameter(NamedTuple):
    target_update_iter: int
    num_episode : int
    save_iter: int
    render: bool
    debug: bool   # if true, disable write result to output_dir
    output_dir: str


class Trainer(ITrainer):
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.loss_durations = []
        self.episode_durations = []
        
    def train(self, param):
        train_start = time.time()
        writer = None if param.debug else SummaryWriter(param.output_dir)
        for episode_i in range(param.num_episode):
            state = self.env.reset()
            #state = torch.from_numpy(state).type(torch.FloatTensor)  # numpy変数をPyTorchのテンソルに変換
            #state = torch.unsqueeze(state, 0)

            start = time.time()
            sum_loss = 0
            for step in count():
                if param.render:
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
                    if not param.debug:
                        writer.add_scalar('reward', step+1, episode_i)
                        writer.add_scalar('training loss',sum_loss/step, episode_i)

                    
                    # 次のエピソードへ
                    break
            # Update the target network, copying all weights and biases in DQN
            if episode_i % param.target_update_iter == 0:
                ''' 目標を修正する '''
                print("Episode: {}, Update Target Net".format(episode_i))
                self.agent.modify_target()

            if episode_i % param.save_iter == param.save_iter-1 and not param.debug:
                fn = "{}/{}.pth".format(param.output_dir, episode_i+1)
                self.agent.record(fn)
                print('Saved model! Name: {}'.format(fn))

        ''' モデルを保存する '''
        if not param.debug:
            # モデルの保存
            fn = "{}/{}.pth".format(param.output_dir, episode_i+1)
            self.agent.record(fn)
            print('Saved model! Name: {}'.format(fn))
        elapsed_time = time.time() - train_start
        print('Completed!, Time: {}'.format(elapsed_time))
        #self.plot_durations()
        
