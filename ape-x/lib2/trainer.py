from abc import *
from typing import NamedTuple, List
import torch
import time
import matplotlib.pyplot as plt
from .replay import IReplay
from .actor import IActor
from .learner import ILearner
import ray
from .util import Color
from .tester import ITester
import numpy as np
import sys
color = Color()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#################################
#####        Trainer       ######
#################################

class ITrainer(metaclass=ABCMeta):
    @abstractmethod
    def train(self):
        '''Q関数の更新'''
        pass

class TrainerParameter(NamedTuple):
    learner: ILearner
    actors : List[IActor]
    replay: IReplay
    tester: ITester
    update_cycles: int
    num_minibatch: int
    batch_size: int

class Trainer(ITrainer):
    def __init__(self, param: TrainerParameter):
        self.learner = param.learner
        self.actors = param.actors
        self.replay = param.replay
        self.tester = param.tester
        self.update_cycles = param.update_cycles
        self.num_minibatch = param.num_minibatch
        self.batch_size = param.batch_size

    def train(self):
        ray.init(ignore_reinit_error=True)
        try:
            start = time.time()

            # learner processの開始
            wip_learner = self.learner.run.remote()

            # actor processの開始
            [actor.run.remote() for actor in self.actors]

            # learner processが終わったらactorを終了させる
            ray.get(wip_learner)
            print("finish leearner")

            wallclocktime = round(time.time() - start, 2)
            color.green("completed {}".format(wallclocktime))
        except KeyboardInterrupt:
            print("shutdown ray processes")
            ray.shutdown()
            sys.exit(1)
