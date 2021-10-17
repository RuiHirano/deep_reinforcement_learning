import time
import sys
sys.path.append('./../')
from lib.replay import ReplayMemory, PrioritizedReplayMemory
from lib.env import Preprocessor, FinanceEnv, FinanceEvaluateEnv
from lib.model import SimpleFinanceNet
from lib.trainer import Trainer, TrainParameter
from lib.examiner import Examiner
from lib.agent import Agent
from lib.brain import Brain, BrainParameter
from lib.util import Color
from backtesting.test import GOOG, EURUSD
from trading_gym_next import EnvParameter, TradingEnv
color = Color()
import yaml
import torch
import json
import datetime
import os
# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#################################
#####         Main         ######
#################################
pp = Preprocessor()
window_size = 10
print(len(GOOG))
test_GOOG = GOOG[1500:]
train_GOOG = GOOG[:1500]
param = EnvParameter(
    df=train_GOOG, 
    mode="random", 
    window_size=window_size+1, # logdiff window is 10
    cash=10000,
    commission=0.01,
    margin=1,
    trade_on_close=False,
    hedging=False,
    exclusive_orders=False,
)
env = FinanceEnv(param)
test_param = EnvParameter(
    df=test_GOOG, 
    mode="random", 
    window_size=window_size+1, # logdiff window is 10
    cash=10000,
    commission=0.01,
    margin=1,
    trade_on_close=False,
    hedging=False,
    exclusive_orders=False,
)
test_env = FinanceEnv(test_param)
num_actions = env.action_space.n
num_states = 1
print(num_actions, num_states)
net = SimpleFinanceNet(num_states, num_actions)

config = {
  "name": "GOOG_simple_lstm_net",
  "debug": False,  # if true, disable write result to output_dir
  "output_dir": "./results/{}".format(datetime.datetime.now().strftime("%Y%m%d%H%M%S")),
  "replay": {
    "type": "PrioritizedExperienceReplay",
    "capacity": 10000
  },
  "brain": {
    "batch_size": 32,
    "gamma": 0.97,
    "eps_start": 0.9,
    "eps_end": 0.05,
    "eps_decay": 200,
    "multi_step_bootstrap": True,
    "num_multi_step_bootstrap": 5,
  },
  "train": {
    "train_mode": True,
    "num_episode": 10000,
    "target_update_iter": 20,
    "render": False,
    "save_iter": 1000
  },
  "eval": {
    "num_episode": 100,
    "filename": "finance_1000.pth",
    "render": True
  }
}

def save_config(output_dir, config):
    ''' configの保存 '''
    fn = "{}/config.yaml".format(output_dir)
    dirname = os.path.dirname(fn)
    if os.path.exists(dirname) == False:
        os.makedirs(dirname)
    with open(fn, "w") as yf:
        yaml.dump(config, yf, default_flow_style=False)

if __name__ == "__main__":
    color.green("device: {}".format(device))

    color.green("config: {}".format(json.dumps(config, indent=2)))
    time.sleep(2)

    ''' ReplayMemory生成 '''
    replay = ReplayMemory(CAPACITY=config["replay"]["capacity"])
    if config["replay"]["type"] == "PrioritizedExperienceReplay":
        replay = PrioritizedReplayMemory(CAPACITY=config["replay"]["capacity"])

    ''' 環境生成 '''
    env = env

    ''' Network生成 '''
    net = net
    
    ''' エージェント生成 '''
    brain_param = BrainParameter(
        replay=replay, 
        net=net, 
        batch_size=config["brain"]["batch_size"],
        gamma=config["brain"]["gamma"],
        eps_start=config["brain"]["eps_start"],
        eps_end=config["brain"]["eps_end"],
        eps_decay=config["brain"]["eps_decay"],
        multi_step_bootstrap=config["brain"]["multi_step_bootstrap"],
        num_multi_step_bootstrap=config["brain"]["num_multi_step_bootstrap"],
    )
    brain = Brain(brain_param, env.action_space.n)
    agent = Agent(brain)

    output_dir = config["output_dir"]
    if not config["debug"]:
        save_config(output_dir, config)

    train_mode = config["train"]["train_mode"]
    if train_mode:
        ''' Trainer '''
        trainer = Trainer(env, test_env, agent)
        train_param = TrainParameter(
            target_update_iter=config["train"]["target_update_iter"],
            num_episode =config["train"]["num_episode"],
            save_iter=config["train"]["save_iter"],
            render=config["train"]["render"],
            debug=config["debug"],
            output_dir=config["output_dir"],
        )
        trainer.train(train_param)
    else:
        print(len(GOOG))
        agent.remember(name="./results/20211002205506/5000.pth")
        ''' Eval '''
        param = EnvParameter(
            df=EURUSD, 
            mode="sequential", 
            window_size=window_size+1, # logdiff window is 10
            cash=10000,
            commission=0.01,
            margin=1,
            trade_on_close=False,
            hedging=False,
            exclusive_orders=False,
        )
        env = FinanceEvaluateEnv(param)
        examiner = Examiner(env, agent)
        examiner.evaluate(config["eval"]["num_episode"], render=False)