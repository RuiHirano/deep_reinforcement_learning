import time
import sys
sys.path.append('./../')
from lib.learner import Learner, LearnerParameter
from lib.replay import Replay, ReplayParameter
from lib.actor import Actor, ActorParameter
from lib.trainer import Trainer, TrainerParameter
from lib.tester import Tester, TesterParameter
from lib.util import Color
from lib.env import CartpoleEnv, BreakoutEnv
from lib.model import DuelingLinearNet, CNNNet, DuelingCNNNet
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

config = {
  "name": "breakout_apex",
  "debug": True,  # if true, disable write result to output_dir
  "train_mode": True,
  "weight_path": "./results/20210919174401/3000.pth",
  "output_dir": "./results/{}".format(datetime.datetime.now().strftime("%Y%m%d%H%M%S")),
  "replay": {
    "capacity": 10000,
    "alpha": 0.6,
    "epsilon": 0.0001
  },
  "actor": {
    "batch_size": 32,
    "gamma": 0.97,
    "eps_start": 0.01,
    "eps_end": 0.5,
    "num_actors": 10,
    "num_rollout": 200,
    "num_multi_step_bootstrap": 5,
  },
  "learner": {
    "batch_size": 32,
    "gamma": 0.97,
    "num_multi_step_bootstrap": 5,
  },
  "train": {
    "num_minibatch": 16,
    "num_update_cycles": 3000,
    "batch_size": 32,
    "save_iter": 300,
  },
  "tester": {
    "num_test_episode": 10,
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

#################################
#####         Main         ######
#################################

if __name__ == "__main__":

    ''' breakout '''
    '''env = BreakoutEnv()
    num_actions = env.action_space.n
    init_screen = env.reset()
    _, ch, screen_height, screen_width = init_screen.shape

    net = DuelingCNNNet(screen_height, screen_width, num_actions)
    optimizer = optim.Adam(net.parameters(), lr=0.001)'''

    '''cartpole'''
    env = CartpoleEnv()
    num_actions = env.action_space.n
    num_states = env.observation_space.shape[0]

    net = DuelingLinearNet(num_states, num_actions)
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    '''actorの作成'''
    num_actors = config["actor"]["num_actors"]
    epsilons = np.linspace(0.01, 0.5, num_actors)
    actors = [Actor.remote(ActorParameter(
        pid=i,
        env=env,
        net=net,
        epsilon=epsilons[i],
        gamma=config["actor"]["gamma"],
        num_multi_step_bootstrap=config["actor"]["num_multi_step_bootstrap"],
        batch_size=config["actor"]["batch_size"],
        num_rollout=config["actor"]["num_rollout"],
    )) for i in range(num_actors)]

    '''replayの作成'''
    replay = Replay(ReplayParameter(
        capacity=config["replay"]["capacity"],
        epsilon=config["replay"]["epsilon"],
        alpha=config["replay"]["alpha"],
    ))

    '''learnerの作成'''
    learner = Learner.remote(LearnerParameter(
        batch_size=config["learner"]["batch_size"],
        gamma=config["learner"]["gamma"],
        num_multi_step_bootstrap=config["learner"]["num_multi_step_bootstrap"],
        net=net,
        optimizer=optimizer,
    ))

    '''testerの作成'''
    tester = Tester.remote(TesterParameter(
        env=env,
        net=net,
        num_test_episode=config["tester"]["num_test_episode"],
        render=False if config["train_mode"] else True
    ))

    train_mode = config["train_mode"]
    if train_mode:
        output_dir = config["output_dir"]
        if not config["debug"]:
            save_config(output_dir, config)

        ''' Trainer '''
        train_param = TrainerParameter(
            learner=learner,
            actors=actors,
            tester=tester,
            replay=replay,
            num_minibatch=config["train"]["num_minibatch"],
            num_update_cycles=config["train"]["num_update_cycles"],
            batch_size=config["train"]["batch_size"],
            save_iter=config["train"]["save_iter"],
            debug=config["debug"],
            output_dir=config["output_dir"],
        )
        trainer = Trainer(train_param)
        trainer.train()
    else:
        color.green("start test")
        weights = torch.load(config["weight_path"])
        wip_tester = tester.test_play.remote(weights)
        mean_test_score = ray.get(wip_tester)
        color.yellow("Score: {}".format(mean_test_score))