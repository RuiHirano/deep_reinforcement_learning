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
    num_update_cycles: int
    num_minibatch: int
    batch_size: int

class Trainer(ITrainer):
    def __init__(self, param: TrainerParameter):
        self.learner = param.learner
        self.actors = param.actors
        self.replay = param.replay
        self.tester = param.tester
        self.num_update_cycles = param.num_update_cycles
        self.num_minibatch = param.num_minibatch
        self.batch_size = param.batch_size

    def train(self):
        ray.init(ignore_reinit_error=True)
        start_total = time.time()
        history = []

        current_weights = ray.get(self.learner.get_current_weights.remote())
        current_weights = ray.put(current_weights)

        wip_actors = [actor.rollout.remote(current_weights) for actor in self.actors]
        
        #: まずはある程度遷移情報を蓄積
        for _ in range(30):
            finished, wip_actors = ray.wait(wip_actors, num_returns=1)
            td_errors, transitions, pid, actor_score = ray.get(finished[0])
            self.replay.push(td_errors, transitions)
            wip_actors.extend([self.actors[pid].rollout.remote(current_weights)])
        
        #: Leanerでのネットワーク更新を開始
        minibatchs = [self.replay.sample(batch_size=self.batch_size) for _ in range(self.num_minibatch)]
        wip_learner = self.learner.update_network.remote(minibatchs)
        wip_tester = self.tester.test_play.remote(current_weights)

        update_cycles = 1
        actor_cycles = 0
        actor_score_all = []
        while update_cycles <= self.num_update_cycles:
            start = time.time()
            actor_cycles += 1
            # actor cycle
            finished, wip_actors = ray.wait(wip_actors, num_returns=1)
            td_errors, transitions, pid, actor_score = ray.get(finished[0])
            #print(actor_score)
            actor_score_all.append(actor_score)
            self.replay.push(td_errors, transitions)
            wip_actors.extend([self.actors[pid].rollout.remote(current_weights)])
            
            #: Learnerのタスク完了判定
            finished_learner, _ = ray.wait([wip_learner], timeout=0)
            if finished_learner:
                current_weights, indices, td_errors, loss = ray.get(finished_learner[0])
                #print("trainer", current_weights['fcA1.weight'][0][0])
                current_weights = ray.put(current_weights)
                #: 優先度の更新とminibatchの作成はlearnerよりも十分に速いという前提
                # replayのpushでのidxの更新とindicesはほぼ被らないという前提
                self.replay.update_priority(indices, td_errors)

                # start next learner cycle
                minibatchs = [self.replay.sample(batch_size=self.batch_size) for _ in range(self.num_minibatch)]
                wip_learner = self.learner.update_network.remote(minibatchs)
                
                mean_actor_score = np.array(actor_score_all).mean()
                actor_score_all = []
                elapsed_time = round(time.time() - start, 2)
                elapsed_total_time = round(time.time() - start_total, 2)
                update_cycles += 1

                if update_cycles % 10 == 0:
                    #:学習状況のtest
                    test_score = ray.get(wip_tester)
                    history.append((update_cycles-5, test_score))
                    wip_tester = self.tester.test_play.remote(current_weights)
                    color.green("Update: {} ({:.1f}%), Loss : {:.8f}, ReplaySize: {}, ActorCycle： {}, ActorScore: {:.1f}, Time: {} sec, {} sec".format(update_cycles, (update_cycles*100 / self.num_update_cycles), loss, len(self.replay), actor_cycles, mean_actor_score, elapsed_time, elapsed_total_time))
                    color.yellow("Test Score: {}".format(test_score))
                
                actor_cycles = 0

        wallclocktime = round(time.time() - start_total, 2)
        cycles, scores = zip(*history)
        plt.plot(cycles, scores)
        plt.title(f"total time: {wallclocktime} sec")
        plt.ylabel("test_score(epsilon=0.01)")
        plt.savefig("history.png")
