from backtesting.test import GOOG
import random
from trading_gym_next import EnvParameter, TradingEnv
from gym import spaces
import gym
import numpy as np
import copy

class CustomEnv(gym.Wrapper):
    def __init__(self, param: EnvParameter):
        self.env = TradingEnv(param)
        self.position_side = 0 # 0: No Position, 1: Long Position, 2: Short Position

    def step(self, action): # action is [0, 1, 2] 0: HOLD, 1: BUY, 2: SELL

        if self.position_side == 1 and action == 1:
            action = 0
        if self.position_side == 2 and action == 2:
            action = 0
        
        obs, reward, done, info = self.env.step(action, size=1)
        pp = Preprocessor()
        logdiff_obs = pp.log_diff(obs)
        #print(logdiff_obs)

        if self.position_side == 1 and action == 2 or self.position_side == 2 and action == 1:
            done = True

        if info["position"].size == 0:
            if self.position_side != 0:
                self.position_side = 0
        elif info["position"].size > 0:
            self.position_side = 1
        elif info["position"].size < 0:
            self.position_side = 2
        
        #print(info["position"], self.position_side, done, action)

        return logdiff_obs, reward, done, info


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
        trade_on_close=True,
        hedging=False,
        exclusive_orders=False,
    )
    env = CustomEnv(param)
    
    for i in range(2):
        print("episode: ", i)
        obs = env.reset()
        done = False
        while not done:
            action = random.choice([0,1,2])
            obs, reward, done, info = env.step(action)
            print("episode: {}, action: {}, reward: {}, done: {}, timestamp: {}, episode_step: {}".format(i, action, reward, done, info["timestamp"], info["episode_step"]))
            print(obs.tail())
    stats = env.stats()
    print(stats)
    #env.plot()