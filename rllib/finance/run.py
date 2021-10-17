import gym, ray
from ray.rllib.agents import ppo, dqn
from trading_gym_next import EnvParameter, TradingEnv
from gym import spaces
from data import USDJPY
from lib.model import SimpleFinanceModel
from lib.env import EnvParameter, FinanceEnv
from ray.rllib.models import ModelCatalog, ActionDistribution
import random


def train():
    ray.init()
    ModelCatalog.register_custom_model("simple_finance_model", SimpleFinanceModel)
    param = EnvParameter(df=USDJPY, mode="sequential", add_feature=True, window_size=100)
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

def eval():
    ray.init()
    
    param = EnvParameter(df=USDJPY, mode="sequential", add_feature=True, window_size=100)
    env = FinanceEnv(param)
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
    env.stats()
    env.plot()

if __name__ == "__main__":
    is_train = True
    if is_train:
        train()
    else:
        eval()