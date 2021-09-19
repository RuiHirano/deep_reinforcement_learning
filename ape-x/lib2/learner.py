from abc import *
from typing import NamedTuple, List
import torch
import torch.nn as nn
import copy
from .replay import Transition, IReplay
from .actor import IActor
from .weights import IWeights
from .tester import ITester
import ray
import time
from .util import Color
color = Color()
# https://horomary.hatenablog.com/entry/2021/03/02/235512#Learner%E3%81%AE%E5%BD%B9%E5%89%B2
# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#################################
#####        Learner         ######
#################################

class ILearner(metaclass=ABCMeta):
    @abstractmethod
    def run(self):
        '''Target Networkの更新'''
        pass

class LearnerParameter(NamedTuple):
    batch_size: int
    gamma : float
    net: nn.Module
    optimizer: torch.optim.Optimizer
    num_multi_step_bootstrap: int  # multi-step bootstrapのステップ数
    replay: IReplay
    weights: IWeights
    tester: ITester
    test_iter: int
    num_update: int

@ray.remote(num_cpus=1)
class Learner(ILearner):
    def __init__(self, param):
        self.batch_size = param.batch_size
        self.gamma = param.gamma
        self.policy_net = param.net.to(device)
        self.target_net = copy.deepcopy(param.net).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = param.optimizer
        self.num_multi_step_bootstrap = param.num_multi_step_bootstrap
        self.replay = param.replay
        self.weights = param.weights
        self.num_update = param.num_update
        self.tester = param.tester
        self.test_iter = param.test_iter

    def run(self):
        time.sleep(3)
        print("start learner process")
        update_count = 0
        start_total = time.time()
        while update_count <= self.num_update:
            #time.sleep(0.05)
            start = time.time()
            if ray.get(self.replay.length.remote()) >= self.batch_size:
                indices, transitions = ray.get(self.replay.sample.remote(self.batch_size), timeout=0)
                update_count += 1
                weights, indices, td_errors, loss = self.update_network(indices, transitions)
                #color.green("weights: {}".format(weights['fcA1.weight'][0][0]))
                weights = ray.put(weights)
                ray.get(self.replay.update_priority.remote(indices, td_errors))
                ray.get(self.weights.set.remote(weights)) # weightsの更新
                replay_size = ray.get(self.replay.length.remote())
                elapsed_time = round(time.time() - start, 2)
                elapsed_total_time = round(time.time() - start_total, 2)
                
                if update_count % self.test_iter == 0:
                    test_score = ray.get(self.tester.test_play.remote(weights))
                    color.green("Update: {} ({:.1f}%), Loss : {:.8f}, ReplaySize: {}, Time: {} sec, {} sec".format(update_count, (update_count*100 / self.num_update), loss, replay_size, elapsed_time, elapsed_total_time))
                    color.yellow("Update: {} ({:.1f}%), TestScore: {}, Time: {} sec, {} sec".format(update_count, (update_count*100 / self.num_update), test_score, elapsed_time, elapsed_total_time))

                #if update_count % self.num_refresh_replay == 0:
                #    color.yellow("refresh replay: {} to 0".format(ray.get(self.replay.length.remote())))
                #    ray.get(self.replay.refresh.remote())
        print("completed learner process")
            

    def update_network(self, indices, transitions):

        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state).to(device) # state: tensor([[0.5, 0.4, 0.5, 0], ...]) size(32, 4)
        action_batch = torch.cat(batch.action).to(device) # action: tensor([[1],[0],[0]...]) size(32, 1) 
        reward_batch = torch.cat(batch.reward).to(device) # reward: tensor([1, 1, 1, 0, ...]) size(32)

        ''' 出力データ：行動価値を作成 '''
        # 出力actionの値のうちaction_batchが選んだ方を抽出（.gather()）
        Q = self.policy_net(state_batch).gather(1, action_batch) # size(32, 1)
        #print(Q.size())
        assert Q.size() == (self.batch_size,1)

        ''' 教師データを作成する '''
        ''' target = 次のステップでの行動価値の最大値 * 時間割引率 + 即時報酬 '''
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool) # non_final_mask: tensor([True, True, True, False, ...]) size(32)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None]).to(device) # size(32-None num,4)
        assert non_final_mask.size() == (self.batch_size,)

        # 次の環境での行動価値
        next_state_values = torch.zeros(self.batch_size, device=device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach() # size(32)
        assert next_state_values.size() == (self.batch_size,)

        # target = rステップ後での行動価値の最大値 * (時間割引率^rステップ) + rステップ後での報酬
        TQ = (next_state_values * (self.gamma ** self.num_multi_step_bootstrap)) + reward_batch
        assert TQ.size() == (self.batch_size,)

        ''' Loss を計算'''
        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        #print(TQ.unsqueeze(1))
        loss = criterion(Q, TQ.unsqueeze(1))

        ''' 勾配計算、更新 '''
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        td_errors = (TQ.unsqueeze(1) - Q).squeeze(1).detach().numpy()
        assert td_errors.shape == (self.batch_size, )

        current_weights = self.policy_net.state_dict()
        self.target_net.load_state_dict(current_weights)
        #print("learner", current_weights['fcA1.weight'][0][0])
        
        return current_weights, indices, td_errors, loss.item()


