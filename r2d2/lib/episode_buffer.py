import collections
from operator import mul
import random
import numpy as np
import torch

Transition = collections.namedtuple(
    "Transition", ["state", "action", "reward", "next_state", "done", "h", "c", "prev_action"])

Segment = collections.namedtuple(
    "Segment", ["states", "actions", "rewards", "dones", "h_init", "c_init", "a_init", "last_state"])


class EpisodeBuffer:

    def __init__(self, burnin_length, unroll_length, gamma, num_multi_step_bootstrap):
        self.transitions = []
        self.burnin_len = burnin_length
        self.unroll_len = unroll_length
        self.gamma = gamma
        self.num_multi_step_bootstrap = num_multi_step_bootstrap

    def __len__(self):
        return len(self.transitions)

    def add(self, transition):
        """
            Optional:
                reward-clipping や n-step-return はここで計算しておくとよい
        """
        #: transition: (s, a, r, s2, done, c, h)
        self.transitions.append(Transition(*transition))

    def pull_segments(self):
        """ 1episode分の遷移を固定長のセグメントに分割する
        """
        segments = []
        multi_step_transitions = self._get_multi_step_transitions(self.transitions)
        if len(multi_step_transitions) >= self.burnin_len + self.unroll_len:
            for t in range(self.burnin_len, len(multi_step_transitions), self.unroll_len):
                if (t + self.unroll_len) > len(multi_step_transitions):
                    #: エピソード終端の長さ修正
                    total_len = self.burnin_len + self.unroll_len
                    timesteps = multi_step_transitions[-total_len:]
                else:
                    timesteps = multi_step_transitions[t-self.burnin_len:t+self.unroll_len]
                batch = Transition(*zip(*timesteps))
                
                #print("deb2", len(self.transitions), len(multi_step_transitions), [x.done for x in multi_step_transitions])
                segment = Segment(
                    states=torch.cat(batch.state), # (timestep, state_size)
                    actions=torch.tensor(batch.action).unsqueeze(1), # (timestep, 1)
                    rewards=torch.tensor(batch.reward[self.burnin_len:]), # (burnin_len, 1)
                    dones=torch.tensor(batch.done[self.burnin_len:]), # (burnin_len, 1)
                    c_init=batch.c[0], # (1, batch_size[1], lstm_out_dim)
                    h_init=batch.h[0], # (1, batch_size[1], lstm_out_dim)
                    a_init=torch.tensor([batch.prev_action[0]]).unsqueeze(0), # (batch_size[1], 1)
                    last_state=batch.next_state[-1] # (batch_size[1], state_size)
                )
                segments.append(segment)
        #print("seg", [x.dones for x in segments])

        return segments

    def _get_multi_step_transitions(self, transitions):
        # 計算用のバッファに遷移を登録
        if len(transitions) < self.num_multi_step_bootstrap:
            return []

        multi_step_transitions = []
        for i, transition in enumerate(transitions):
            nstep_reward = 0
            for t in range(self.num_multi_step_bootstrap):
                if i+t < len(transitions):
                    tr = transitions[i+t]
                    r = tr.reward
                    nstep_reward += r * self.gamma ** t

                    # 終端の場合、それ以降の遷移は次のepisodeのものなので計算しない
                    if tr.next_state is None:
                        break

            # rewardを更新
            transition = Transition(transition.state, transition.action, nstep_reward, transition.next_state, transition.done, transition.h, transition.c, transition.prev_action)
    
            # 時刻tでのstateとaction、t+nでのstate、その間での報酬の割引累積和をreplay memoryに登録
            multi_step_transitions.append(transition)
        return multi_step_transitions

