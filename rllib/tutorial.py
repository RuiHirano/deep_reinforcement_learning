import gym, ray
from ray.rllib.agents import ppo, dqn
from trading_gym_next import EnvParameter, TradingEnv
from gym import spaces
from backtesting.test import GOOG, EURUSD
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import random
from ray.rllib.models import ModelCatalog, ActionDistribution
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune.registry import register_env

#################################
#####         Net          ######
#################################

class SimpleFinanceModel(TorchModelV2):
    '''線形入力でDuelingNetworkを搭載したDQN'''
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        print(obs_space.shape[1], action_space.n)
        self.num_states = obs_space.shape[1]
        self.num_actions = action_space.n
        self.lstm = nn.LSTM(self.num_states, 32, batch_first=True)
        self.dense = nn.Linear(32, self.num_actions)

    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"] # (32, 10, 4) N, L, H
        x, _ = self.lstm(x)
        x = F.relu(x[:, -1, :])
        x = self.dense(x)
        return x, []
ModelCatalog.register_custom_model("simple_finance_model", SimpleFinanceModel)

class FinanceActionDist(ActionDistribution):
    @staticmethod
    def required_model_output_shape(action_space, model_config):
        return 3  # controls model output feature vector size

    def __init__(self, inputs, model):
        super(FinanceActionDist, self).__init__(inputs, model)
        assert model.num_outputs == 3
ModelCatalog.register_custom_action_dist("finance_dist", FinanceActionDist)

#################################
#####         Env          ######
#################################

class FinanceEnv(gym.Wrapper):
    def __init__(self, env_config):
        param = env_config["param"]
        param.window_size += 1 # because drop by log_diff
        super().__init__(TradingEnv(param))
        self.action_space = spaces.Discrete(3) # action is [0, 1, 2] 0: HOLD, 1: BUY, 2: SELL
        self.observation_space = spaces.Box(low=-10, high=10, shape=(param.window_size-1, 4), dtype=np.float64)
        self.position_side = 0 # 0: No Position, 1: Long Position, 2: Short Position
        self.step_num = 0

    def step(self, action):
        if self.position_side == 1 and action == 1:
            action = 0
        if self.position_side == 2 and action == 2:
            action = 0
        
        if self.position_side != 0:
            self.step_num += 1
        if self.step_num >= 20:
            if self.position_side == 1:
                action = 2
            elif self.position_side == 2:
                action = 1

        obs, reward, done, info = self.env.step(action, size=1)
        logdiff = self.log_diff(obs.loc[:,['Close','Volume']])
        obs = pd.merge(logdiff, obs.loc[:,['Position','Reward']], left_index=True, right_index=True)
        if (self.position_side == 1 and action == 2) or (self.position_side == 2 and action == 1) or self.step_num >= 20:
            done = True

        if info["position"].size > 0:
            self.position_side = 1
        elif info["position"].size < 0:
            self.position_side = 2
        return obs.to_numpy(), reward, done, info
    
    def log_diff(self, df):
        return np.log(df).diff(1)[1:]

    def reset(self):
        self.step_num = 0
        self.position_side = 0
        obs = self.env.reset()
        logdiff = self.log_diff(obs.loc[:,['Close','Volume']])
        obs = pd.merge(logdiff, obs.loc[:,['Position','Reward']], left_index=True, right_index=True)
        return obs.to_numpy()

class CustomEnv(gym.Wrapper):
    def __init__(self, env_config):
        env = gym.make('CartPole-v0')
        super().__init__(env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        print(obs)
        return obs, reward, done, info

def run_env():
    param = EnvParameter(df=GOOG[:200], mode="sequential", add_feature=True, window_size=100)
    print(GOOG)
    env = FinanceEnv(env_config={"param": param})
    
    for i in range(20):
        print("episode: ", i)
        obs = env.reset()
        done = False
        while not done:
            action = random.choice([0,1,2])
            next_obs, reward, done, info = env.step(action)
            print("obs", obs[-5:])
            print("action: ", action)
            print("date: {}, reward: {}, done: {}, timestamp: {}, episode_step: {}, position: {}".format(info["date"], reward, done, info["timestamp"], info["episode_step"], info["position"]))
            print("next_obs", next_obs[-5:])
            print("-"*10)
            obs = next_obs
    print("finished")
    stats = env.stats()
    print(stats)
#run_env()

def train():
    ray.init()
    param = EnvParameter(df=EURUSD, mode="sequential", add_feature=True, window_size=100)
    trainer = dqn.DQNTrainer(env=FinanceEnv, config={
        "env_config": {"param": param},  # config to pass to env class
        "model": {
            "custom_model": "simple_finance_model",
            "custom_model_config": {},
            "custom_action_dist": "finance_dist",
            "fcnet_hiddens": [3, 3],
            "fcnet_activation": "relu",
        },
        "framework": "torch",
    })

    while True:
        print(trainer.train())
train()

def eval():
    ray.init()
    param = EnvParameter(df=EURUSD, mode="sequential", add_feature=True, window_size=100)
    env = FinanceEnv(param)
    obs = env.reset()
    done = False
    #model = 
    while not done:
        action = random.choice([0,1,2])
        next_obs, reward, done, info = env.step(action)
        print("obs", obs[-5:])
        print("action: ", action)
        print("date: {}, reward: {}, done: {}, timestamp: {}, episode_step: {}, position: {}".format(info["date"], reward, done, info["timestamp"], info["episode_step"], info["position"]))
        print("next_obs", next_obs[-5:])
        print("-"*10)
        obs = next_obs
    env.stats()
    env.plot()
#eval()