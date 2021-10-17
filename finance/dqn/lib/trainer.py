from abc import *
from typing import NamedTuple
import torch
from itertools import count
import time
from .replay import Transition
from .util import Color
import copy
from torch.utils.tensorboard import SummaryWriter

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
color = Color()
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
    def __init__(self, env, test_env, agent):
        self.env = env
        self.test_env = test_env
        self.agent = agent
        self.loss_durations = []
        self.episode_durations = []

    def change_state(self, state):
        '''change state Tensor(L, N, Hin)'''
        state = torch.tensor(state.values).float() # (10, 5)
        # extract Close data
        state = state[:, 3].unsqueeze(1) # OHLCV 3 is Close (10, 1)
        state = state.unsqueeze(1) # add batch (10, 1, 1) (L, B, Hin)
        return state
        
    def train(self, param):
        train_start = time.time()
        writer = None if param.debug else SummaryWriter(param.output_dir)
        for episode_i in range(param.num_episode):
            state = self.env.reset()
            state = self.change_state(state)

            start = time.time()
            sum_loss = 0
            reward_all = 0
            for step in count():
                if param.render:
                    self.env.render()

                ''' 行動を決定する '''
                action = self.agent.select_action(state) # input ex: <list> [0, 0, 0, 0], output ex: <int> 0 or 1
                ''' 行動に対する環境や報酬を取得する '''
                next_state, reward, done, info = self.env.step(action)  # state [0,0,0,0...window_size], reward 1.0, done False, input: action 0 or 1 or 2
                next_state = self.change_state(next_state)
                ''' エージェントに記憶させる '''
                self.agent.memorize(
                    Transition(
                        state, 
                        action, 
                        next_state, 
                        reward,
                        done
                    )
                )

                # Move to the next state
                state = copy.deepcopy(next_state)

                ''' エージェントに学習させる '''
                reward_all += reward
                loss = self.agent.learn()
                if loss != None:
                    sum_loss += loss

                if done:
                    elapsed_time = time.time() - start
                    ''' 終了時に結果をプロット '''
                    #print(loss, episode_i)
                    if reward_all > 0:
                        color.green("Episode: {}, Step: {}, Loss: {:.5f}, Reward: {:.5f}, Position: {}, Time: {:.2f} [sec]".format(episode_i, step+1, sum_loss/(step+1), reward_all/(step+1), info["position"], elapsed_time))
                    else:
                        color.red("Episode: {}, Step: {}, Loss: {:.5f}, Reward: {:.5f}, Position: {}, Time: {:.2f} [sec]".format(episode_i, step+1, sum_loss/(step+1), reward_all/(step+1), info["position"], elapsed_time))
                    #self.episode_durations.append(step+1 + 1)
                    if not param.debug:
                        writer.add_scalar('reward', reward_all/(step+1), episode_i)
                        writer.add_scalar('training loss',sum_loss/(step+1), episode_i)

                    
                    # 次のエピソードへ
                    break
            # Update the target network, copying all weights and biases in DQN
            if episode_i % param.target_update_iter == 0:
                ''' 目標を修正する '''
                print("Episode: {}, Update Target Net".format(episode_i))
                self.agent.modify_target()
                # test
                self.test()

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
        
    def test(self):
        print("test")
        episode_num = 10
        score_all = []
        step_all = []
        start = time.time()
        for i in range(episode_num):
            done = False
            step = 0
            score = 0
            state = self.test_env.reset()
            state = self.change_state(state)
            while not done:
                step += 1
                #print(step)
                ''' 行動を決定する '''
                action = self.agent.predict_action(state)
                ''' 行動に対する環境や報酬を取得する '''
                next_state, reward, done, _ = self.test_env.step(action)  # state [0,0,0,0...window_size], reward 1.0, done False, input: action 0 or 1 or 2
                next_state = self.change_state(next_state)
                state = next_state
                score += reward
            score_all.append(score/step)
            step_all.append(step)
        elapsed_time = time.time() - start
        print("Test: Step: {}, Reward: {:.5f}, Time: {:.2f} [sec]".format(sum(step_all)/episode_num, sum(score_all)/episode_num, elapsed_time))
