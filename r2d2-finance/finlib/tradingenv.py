import numpy as np
from backtesting import Backtest, Strategy
from backtesting._stats import compute_stats
from abc import *
import gym
from backtesting.test import SMA, GOOG
import time
from gym import spaces
from threading import (Event, Thread)
import random
from typing import NamedTuple
import pandas as pd
import sys
import threading

class TradingStrategy(Strategy):
    def init(self):
        pass

    def next(self):
        self.callback(self)

class BacktestingThread(threading.Thread):
    def __init__(self, data):
        threading.Thread.__init__(self)
        TradingStrategy.callback = self._callback
        self.bt = Backtest(data, TradingStrategy, commission=.002)
        #print(self.bt._cash)
        self.strategy = None
        self.result = None
        self.event = Event()
        self.event2 = Event()
        self.kill_event = Event()
        self.count = 0
        
    def run(self):
        self.result = self.bt.run()

    def get(self):
        #print(self.strategy._data.df.tail(1))
        if not self.kill_event.is_set():
            self.event.set()
            self.event2.wait()
            self.event2.clear()
        return self.strategy

    def kill(self):
        print("killed")
        self.event.set()
        self.event2.set()
        self.kill_event.set()

    def _callback(self, strategy: Strategy):
        self.strategy = strategy
        if not self.kill_event.is_set():
            self.event2.set()
            self.event.wait()
            self.event.clear()

    def result(self):
        return self.result

class EnvParameter:
    def __init__(self, df: pd.DataFrame, window_size: int,
        episode_length: int, mode: str = "sequential", step_length: int = 1
    ):
        self.df = df
        self.window_size = window_size
        self.episode_length = episode_length  
        self.step_length = step_length
        self.mode = mode  # "sequential": sequential episode, "random": episode start is random
        self._check_param()

    def _check_param(self):
        self._check_column()
        self._check_length()
        self._check_mode()

    def _check_column(self):
        # column check
        if not all([item in self.df.columns for item in ['Open', 'High', 'Low', 'Close']]):
            raise RuntimeError(("Required column is not exist."))

    def _check_length(self):
        if self.window_size + (self.episode_length - 1) > len(self.df):
            raise RuntimeError("df length is not enough.")

    def _check_mode(self):
        if self.mode not in ["sequential", "random"]:
            raise RuntimeError(("Parameter mode is invalid. You should be 'random' or 'sequential'"))



class TradingEnv(gym.Wrapper):
    def __init__(self, param: EnvParameter):
        self.param = param
        self.df = param.df
        self.strategy = None
        self.finished = False
        self.mode = param.mode  # "sequential": sequential episode, "random": episode start is random
        self.timestamp = 0
        self.episode_length = param.episode_length
        self.step_length = 1
        self.window_size = param.window_size
        self.count = 0
        self.action_space = spaces.Discrete(3) # action is [0, 1, 2] 0: HOLD, 1: BUY, 2: SELL
        self.event = Event()
        self.event2 = Event()
        self.run_backtesting(self.df)

    def _callback(self, strategy: Strategy):
        self.strategy = strategy
        if len(strategy.data) >= self.window_size:
            self.event.wait()
            self.event.clear()
            self.event2.set()
        if self.finished:
            sys.exit(1)

    def buy(self):
        self.strategy.buy()

    def sell(self):
        self.strategy.sell()

    def step(self, action):
        self.event.set()
        self.event2.wait()
        self.event2.clear()

        if action == 1:
            self.buy()
        elif action == 2:
            self.sell()
        self.count += 1

        obs = self.get_observation()
        reward = self.get_reward()
        done = self.get_done()
        info = self.get_info()
        self.forward_timestamp()
        return obs, reward, done, info

    def reset(self):
        self.set_next_episode_timestamp()
        data = self.get_next_episode_data()
        self.run_backtesting(data)
        obs = self.get_observation()
        return obs

    def forward_timestamp(self):
        self.timestamp += self.step_length

    def get_observation(self):
        return self.strategy.data.df[-self.window_size:]

    def get_reward(self):
        # sum of profit percentage
        return sum([trade.pl_pct for trade in self.strategy.trades])

    def get_done(self):
        if len(self.strategy.data.df) >= len(self.df):
            return True
        return False

    def get_info(self):
        return {
            "timestamp": self.timestamp,
            "orders": self.strategy.orders, 
            "trades": self.strategy.trades, 
            "position": self.strategy.position, 
            "closed_trades": self.strategy.closed_trades, 
        }

    def run_backtesting(self, data):
        TradingStrategy.callback = self._callback
        self.bt = Backtest(data, TradingStrategy, commission=.002, exclusive_orders=True)
        self.thread = Thread(target=self.bt.run)
        self.thread.start()
        time.sleep(0.1)

    def set_next_episode_timestamp(self):
        if self.mode == "random":
            self.timestamp = random.choice(range(len(self.df)))
        elif self.mode == "sequential":
            if self.timestamp + self.window_size > len(self.df):
                self.timestamp = 0
        return self.timestamp
    
    def get_next_episode_data(self):
        data = self.df[self.timestamp:]
        return data
    
if __name__ == "__main__":
    '''param = EnvParameter(df=GOOG[:90], mode="random", window_size=10, episode_length=10)
    env = TradingEnv(param)
    
    for k in range(5):
        obs = env.reset()
        print(k, obs)
        for i in range(10):
            action = random.choice([0,1,2])
            obs, reward, done, info = env.step(action)
            print(reward, done, info["timestamp"])'''

    bt = BacktestingThread(GOOG)
    bt.start()
    for i in range(10):
        st = bt.get()
        st.buy(size=i+1)
        print(st.data.df.tail(1))
        print(st._broker._cash, st.position, st.orders, st.trades, st.closed_trades)
        print(sum([trade.pl_pct for trade in st.trades]))
    bt.kill()