from backtesting.test import GOOG
import random
from trading_gym_next import EnvParameter, TradingEnv
from gym import spaces
import gym
import numpy as np
import copy

class FinanceEnv(gym.Wrapper):
    def __init__(self, param: EnvParameter):
        env = TradingEnv(param)
        super().__init__(env)
        self.position_side = 0 # 0: No Position, 1: Long Position, 2: Short Position
        self.step_num = 0

    def step(self, action):
        self.step_num += 1
        if self.position_side == 1 and action == 1:
            action = 0
        if self.position_side == 2 and action == 2:
            action = 0

        obs, reward, done, info = self.env.step(action, size=1)
        pp = Preprocessor()
        logdiff_obs = pp.log_diff(obs)
        if (self.position_side == 1 and action == 2) or (self.position_side == 2 and action == 1) or self.step_num > 20:
            done = True

        if info["position"].size == 0:
            if self.position_side != 0:
                self.position_side = 0
        elif info["position"].size > 0:
            self.position_side = 1
        elif info["position"].size < 0:
            self.position_side = 2
        return logdiff_obs, reward, done, info

    def reset(self):
        self.step_num = 0
        obs = self.env.reset()
        pp = Preprocessor()
        logdiff_obs = pp.log_diff(obs)
        return logdiff_obs

class FinanceEvaluateEnv(gym.Wrapper):
    def __init__(self, param: EnvParameter):
        env = TradingEnv(param)
        super().__init__(env)
        self.env = env
        self.position_side = 0 # 0: No Position, 1: Long Position, 2: Short Position

    def step(self, action):

        if self.position_side == 1 and action == 1:
            action = 0
        if self.position_side == 2 and action == 2:
            action = 0

        obs, reward, done, info = self.env.step(action, size=1)
        pp = Preprocessor()
        logdiff_obs = pp.log_diff(obs)

        if info["position"].size == 0:
            if self.position_side != 0:
                self.position_side = 0
        elif info["position"].size > 0:
            self.position_side = 1
        elif info["position"].size < 0:
            self.position_side = 2
        return logdiff_obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        pp = Preprocessor()
        logdiff_obs = pp.log_diff(obs)
        return logdiff_obs

class Preprocessor():
    def __init__(self):
        pass

    def log_diff(self, df):
        return np.log(df).diff(1)[1:]

if __name__ == "__main__":
    pp = Preprocessor()
    log_GOOG = pp.log_diff(GOOG)

    window_size = 10
    param = EnvParameter(
        df=GOOG[:40], 
        mode="sequential", 
        window_size=window_size+1, # logdiff window is 10
        cash=10000,
        commission=0.01,
        margin=1,
        trade_on_close=False,
        hedging=False,
        exclusive_orders=False,
    )
    env = FinanceEnv(param)
    print("test", env.observation_space, env.action_space)
    
    for i in range(2):
        print("episode: ", i)
        obs = env.reset()
        done = False
        while not done:
            action = random.choice([0,1,2])
            obs, reward, done, info = env.step(action)
            print("episode: {}, action: {}, reward: {}, done: {}, timestamp: {}, episode_step: {}, position: {}".format(i, action, reward, done, info["timestamp"], info["episode_step"], info["position"]))
            print(obs)
    stats = env.stats()
    print(stats)
    #env.plot()