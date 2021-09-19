import time
import sys
sys.path.append('./../')
from lib.learner import Learner, LearnerParameter
from lib.replay import Replay, ReplayParameter
from lib.actor import Actor, ActorParameter
from lib.trainer import Trainer, TrainerParameter
from lib.tester import Tester, TesterParameter
from lib.util import Color
import torch.optim as optim
color = Color()
import yaml
from pathlib import Path
import torch
import torch.nn as nn
import gym
import numpy as np
import datetime
import os
import ray
# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#################################
#####         Net          ######
#################################

class DuelingDQN(nn.Module):
    '''線形入力でDualingNetworkを搭載したDQN'''
    def __init__(self, num_states, num_actions):
        super(DuelingDQN, self).__init__()
        self.num_states = num_states
        self.num_actions = num_actions

        self.fc1 = nn.Linear(self.num_states, 32)
        self.relu = nn.ReLU()
        self.fcV1 = nn.Linear(32, 32)
        self.fcA1 = nn.Linear(32, 32)
        self.fcV2 = nn.Linear(32, 1)
        self.fcA2 = nn.Linear(32, self.num_actions)

    def forward(self, x):
        x = self.relu(self.fc1(x))

        V = self.fcV2(self.fcV1(x))
        A = self.fcA2(self.fcA1(x))

        averageA = A.mean(1).unsqueeze(1)
        return V.expand(-1, self.num_actions) + (A - averageA.expand(-1, self.num_actions))

#################################
#####      Environment     ######
#################################

class CartpoleEnv(gym.Wrapper):
    def __init__(self):
        env = gym.make('CartPole-v0').unwrapped
        gym.Wrapper.__init__(self, env)
        self.episode_step = 0
        self.complete_episodes = 0
        
    def step(self, action): 
        observation, reward, done, info = self.env.step(action)
        self.episode_step += 1

        state = torch.from_numpy(observation).type(torch.FloatTensor)  # numpy変数をPyTorchのテンソルに変換
        state = torch.unsqueeze(state, 0)

        if self.episode_step == 200: # 200以上でdoneにする
            done = True

        if done:
            state = None
            if self.episode_step > 195:
                reward = 1
                self.complete_episodes += 1  # 連続記録を更新
                #if self.complete_episodes >= 10:
                #    print("{}回連続成功".format(self.complete_episodes))
            else:
                # こけたら-1を与える
                reward = -1
                self.complete_episodes = 0
            
            self.episode_step = 0

        return state, reward, done, info

    def reset(self):
        observation = self.env.reset()
        state = torch.from_numpy(observation).type(torch.FloatTensor)  # numpy変数をPyTorchのテンソルに変換
        state = torch.unsqueeze(state, 0)
        return state

#################################
#####         Main         ######
#################################

def save_config(name: str, config):
    id = datetime.datetime.now().strftime("%Y%m%d%H%M%S") + name

    ''' configの保存 '''
    fn = "./results/{}/config.yaml".format(id)
    dirname = os.path.dirname(fn)
    if os.path.exists(dirname) == False:
        os.makedirs(dirname)
    with open(fn, "w") as yf:
        yaml.dump(config, yf, default_flow_style=False)

def load_config(filename: str):
    with open("{}".format(Path(filename).resolve()), 'r') as f:
        d = yaml.safe_load(f)
    return d

if __name__ == "__main__":
    #parser = argparse.ArgumentParser()
    #parser.add_argument('-f', '--file',type=str, default='', help='Config file')
    #args = parser.parse_args()
    #color.green("device: {}".format(device))

    #if args.file == "":
    #    color.red("Config is not found: Please use -f option")
    #    sys.exit(1)

    #'''configの読み取り'''
    #config = load_config(args.file)
    #color.green("config: {}".format(json.dumps(config, indent=2)))
    #time.sleep(2)

    #''' configの保存 '''
    #name = config["info"]["name"]
    #save_config(name)

    '''envの作成'''
    env = CartpoleEnv()
    num_actions = env.action_space.n
    num_states = env.observation_space.shape[0]

    '''netの作成'''
    net = DuelingDQN(num_states, num_actions)

    '''actorの作成'''
    num_actors = 10
    epsilons = np.linspace(0.01, 0.5, num_actors)
    actors = [Actor.remote(ActorParameter(
        pid=i,
        env=env,
        net=net,
        epsilon=epsilons[i],
        gamma=0.98,
        num_multi_step_bootstrap=5,
        batch_size=32,
        num_rollout=200,
    )) for i in range(num_actors)]

    '''replayの作成'''
    replay = Replay(ReplayParameter(
        capacity=10000,
        epsilon=0.0001,
        alpha=0.6
    ))

    '''learnerの作成'''
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    learner = Learner.remote(LearnerParameter(
        batch_size=32,
        gamma=0.98,
        net=net,
        optimizer=optimizer,
        num_multi_step_bootstrap=5
    ))

    '''testerの作成'''
    tester = Tester.remote(TesterParameter(
        env=env,
        net=net,
        num_test_episode=10,
    ))

    train_mode = True
    if train_mode:
        ''' Trainer '''
        train_param = TrainerParameter(
            learner=learner,
            actors=actors,
            tester=tester,
            replay=replay,
            num_minibatch=16,
            num_update_cycles=3000,
            batch_size=32,
        )
        trainer = Trainer(train_param)
        trainer.train()
    else:
        pass