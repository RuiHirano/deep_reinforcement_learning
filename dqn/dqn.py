import time
import sys
sys.path.append('./../')
from lib.dqn import Trainer, Examiner, Brain, Agent, BrainParameter, TrainParameter
from lib.replay_memory import ReplayMemory, PrioritizedReplayMemory
from lib.util import Color
color = Color()
import yaml
from pathlib import Path
import torch
import torch.nn as nn
import gym
import argparse
from importlib import import_module
import json
import datetime
import os
# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#################################
#####         Main         ######
#################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file',type=str, default='', help='Config file')
    args = parser.parse_args()
    color.green("device: {}".format(device))

    if args.file == "":
        color.red("Config is not found: Please use -f option")
        sys.exit(1)

    def load_yaml(filename: str):
        with open("{}".format(Path(filename).resolve()), 'r') as f:
            d = yaml.safe_load(f)
        return d
    config = load_yaml(args.file)
    color.green("config: {}".format(json.dumps(config, indent=2)))
    time.sleep(2)

    ''' Memory生成 '''
    memory = ReplayMemory(CAPACITY=config["replay"]["capacity"])
    if config["replay"]["type"] == "PrioritizedExperienceReplay":
        memory = PrioritizedReplayMemory(CAPACITY=config["replay"]["capacity"])

    ''' 環境生成 '''
    module = import_module("data.{}".format(config["info"]["module_name"]))
    env, net = module.get_env_net()
    
    ''' エージェント生成 '''
    brain_param = BrainParameter(
        replay_memory=memory, 
        net=net, 
        batch_size=config["train"]["batch_size"],
        gamma=config["train"]["gamma"],
        eps_start=config["train"]["eps_start"],
        eps_end=config["train"]["eps_end"],
        eps_decay=config["train"]["eps_decay"],
        multi_step_bootstrap=config["train"]["multi_step_bootstrap"],
        num_multi_step_bootstrap=config["train"]["num_multi_step_bootstrap"],
    )
    brain = Brain(brain_param, env.action_space.n)
    agent = Agent(brain)

    name = config["info"]["name"]
    id = name+datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    ''' configの保存 '''
    fn = "./results/{}/config.yaml".format(id)
    dirname = os.path.dirname(fn)
    if os.path.exists(dirname) == False:
        os.makedirs(dirname)
    with open(fn, "w") as yf:
        yaml.dump(config, yf, default_flow_style=False)

    train_mode = config["train"]["train_mode"]
    if train_mode:
        ''' Trainer '''
        trainer = Trainer(id, env, agent)
        train_param = TrainParameter(
            target_update_iter=config["train"]["target_update_iter"],
            num_episode =config["train"]["num_episode"],
            save_iter=config["train"]["save_iter"],
            save_filename=config["train"]["save_filename"],
            render=config["train"]["render"],
        )
        trainer.train(train_param)
    else:
        agent.remember(name="./results/cartpole/20210912_dualingdqn2/cartpole_6000.pth")
        ''' Eval '''
        examiner = Examiner(env, agent)
        examiner.evaluate(config["eval"]["num_episode"], render=config["eval"]["render"])