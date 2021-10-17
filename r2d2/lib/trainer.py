import sys
sys.path.append('./../')
from abc import *
from typing import NamedTuple, List
import torch
import time
import matplotlib.pyplot as plt
from lib.replay import IReplay
from lib.actor import IActor
from lib.learner import ILearner
import ray
from lib.util import Color
from lib.tester import ITester
import numpy as np
from torch.utils.tensorboard import SummaryWriter
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
    save_iter: int
    debug: bool   # if true, disable write result to output_dir
    output_dir: str


class Trainer(ITrainer):
    def __init__(self, param: TrainerParameter):
        self.learner = param.learner
        self.actors = param.actors
        self.replay = param.replay
        self.tester = param.tester
        self.num_update_cycles = param.num_update_cycles
        self.num_minibatch = param.num_minibatch
        self.batch_size = param.batch_size
        self.save_iter = param.save_iter
        self.debug = param.debug
        self.output_dir = param.output_dir
        self.writer = None if self.debug else SummaryWriter(self.output_dir)

    def train(self):
        ray.init(ignore_reinit_error=True)
        start_total = time.time()

        current_weights = ray.get(self.learner.get_current_weights.remote())
        current_weights = ray.put(current_weights)

        wip_actors = [actor.rollout.remote(current_weights) for actor in self.actors]
        
        #: まずはある程度遷移情報を蓄積
        color.green("Initial actor cycle")
        for _ in range(30):
            finished, wip_actors = ray.wait(wip_actors, num_returns=1)
            priorities, segments, pid, actor_score = ray.get(finished[0])
            self.replay.push(priorities, segments)
            wip_actors.extend([self.actors[pid].rollout.remote(current_weights)])
        
        #: Leanerでのネットワーク更新を開始
        minibatchs = [self.replay.sample(batch_size=self.batch_size) for _ in range(self.num_minibatch)]
        wip_learner = self.learner.update_network.remote(minibatchs)
        wip_tester = self.tester.test_play.remote(current_weights)

        update_cycles = 1
        test_update_cycles = 1 # test実行時のupdate_cycles
        actor_cycles = 0
        actor_score_all = []

        color.green("Start main cycles")
        while update_cycles <= self.num_update_cycles:
            start = time.time()
            actor_cycles += 1
            # actor cycle
            finished, wip_actors = ray.wait(wip_actors, num_returns=1)
            priorities, segments, pid, actor_score = ray.get(finished[0])
            #print(actor_score)
            actor_score_all.append(actor_score)
            self.replay.push(priorities, segments)
            wip_actors.extend([self.actors[pid].rollout.remote(current_weights)])
            
            #: Learnerのタスク完了判定
            finished_learner, _ = ray.wait([wip_learner], timeout=0)
            if finished_learner:
                current_weights, indices, priorities, loss = ray.get(finished_learner[0])
                current_weights = ray.put(current_weights)
                #: 優先度の更新とminibatchの作成はlearnerよりも十分に速いという前提
                # replayのpushでのidxの更新とindicesはほぼ被らないという前提
                self.replay.update_priority(indices, priorities)

                # start next learner cycle
                minibatchs = [self.replay.sample(batch_size=self.batch_size) for _ in range(self.num_minibatch)]
                wip_learner = self.learner.update_network.remote(minibatchs)
                
                mean_actor_score = np.array(actor_score_all).mean()
                elapsed_time = round(time.time() - start, 2)
                elapsed_total_time = round(time.time() - start_total, 2)
                update_cycles += 1
                
                if update_cycles % 10 == 0:
                    #:学習状況のtest
                    test_update_cycles = update_cycles
                    wip_tester = self.tester.test_play.remote(current_weights)
                    color.green("Update: {} ({:.1f}%), Loss : {:.8f}, ReplaySize: {}, ActorCycle： {}, ActorScore: {:.1f}, Time: {} sec, {} sec".format(update_cycles, (update_cycles*100 / self.num_update_cycles), loss, len(self.replay), actor_cycles, mean_actor_score, elapsed_time, elapsed_total_time))
                    print("actor scores", actor_score_all[:10])
                    if not self.debug:
                        #self.writer.add_scalar('test_score', test_score, update_cycles)
                        self.writer.add_scalar('loss', loss, update_cycles)

                if update_cycles % self.save_iter == self.save_iter-1 and not self.debug:
                    fn = "{}/{}.pth".format(self.output_dir, update_cycles+1)
                    self.learner.save_model.remote(fn)
                    print('Saved model! Name: {}'.format(fn))
                actor_score_all = []
                actor_cycles = 0

            # test終了時の処理
            if wip_tester != None:
                finished_tester, _ = ray.wait([wip_tester], timeout=0)
                if finished_tester:
                    wip_tester = None
                    test_score = ray.get(finished_tester[0])
                    color.yellow("Update: {}, Test Score: {}".format(test_update_cycles, test_score))
                    if not self.debug:
                        self.writer.add_scalar('test_score', test_score, update_cycles)


        ''' モデルを保存する '''
        if not self.debug:
            # モデルの保存
            fn = "{}/{}.pth".format(self.output_dir, update_cycles+1)
            self.learner.save_model.remote(fn)
            print('Saved model! Name: {}'.format(fn))
        elapsed_time = time.time() - start_total
        print('Completed!, Time: {} sec'.format(elapsed_time))
