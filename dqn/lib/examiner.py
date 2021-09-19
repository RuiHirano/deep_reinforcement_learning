from abc import *
from itertools import count
import time
import matplotlib.pyplot as plt

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