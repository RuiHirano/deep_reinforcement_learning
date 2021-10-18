import gym, ray
from ray.rllib.agents import ppo, dqn
from trading_gym_next import EnvParameter, TradingEnv
from gym import spaces
from data import USDJPY
from lib.model import RLRibSimpleFinanceModel, SimpleFinanceModel
from lib.env import EnvParameter, FinanceEnv
from ray.rllib.models import ModelCatalog, ActionDistribution
import random
import torch
import os
from datetime import datetime
from ray.tune.logger import UnifiedLogger

def custom_log_creator():
    cwd = os.getcwd()
    timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    dirpath = "{}/results/{}_{}".format(cwd, "DQN_FinanceEnv", timestr)

    def logger_creator(config):

        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        return UnifiedLogger(config, dirpath, loggers=None)

    return logger_creator

def train():
    ray.init()
    ModelCatalog.register_custom_model("simple_finance_model", RLRibSimpleFinanceModel)
    param = EnvParameter(df=USDJPY, mode="random", add_feature=True, window_size=100)
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
    }, logger_creator=custom_log_creator())

    for i in range(10000):
        print(trainer.train())
        if i % 100 == 0:
            checkpoint = trainer.save()
            print("checkpoint saved at", checkpoint)

def eval():
    ray.init()
    
    param = EnvParameter(df=USDJPY[:2000], mode="sequential", add_feature=True, window_size=100)
    env = FinanceEnv(env_config={"param": param})
    
    ModelCatalog.register_custom_model("simple_finance_model", RLRibSimpleFinanceModel)
    path = "/Users/ruihirano/ray_results/DQN_FinanceEnv_2021-10-17_23-59-15dsshcsl2/checkpoint_000401/checkpoint-401"
    agent = dqn.DQNTrainer(env=FinanceEnv, config={
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
    agent.restore(path)

    obs = env.reset()
    step = 0
    while step < 2000:
        step += 1
        action = agent.compute_action(obs)
        next_obs, reward, done, info = env.step(action)
        #print("obs", obs[-5:])
        #print("action: ", action)
        print("date: {}, reward: {}, done: {}, timestamp: {}, episode_step: {}, position: {}".format(info["date"], reward, done, info["timestamp"], info["episode_step"], info["position"]))
        #print("next_obs", next_obs[-5:])
        #print("-"*10)
        obs = next_obs
    env.env.stats()
    env.env.plot()

if __name__ == "__main__":
    is_train = True
    train_data = USDJPY[:int(len(USDJPY)*0.8)] # 379910
    eval_data = USDJPY[int(len(USDJPY)*0.8):] # 94980
    print(train_data.size, eval_data.size)
    if is_train:
        train()
    else:
        eval()