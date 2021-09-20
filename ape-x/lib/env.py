from abc import *
import torch
import gym
from typing import NamedTuple
import torchvision.transforms as T
import numpy as np
from gym import spaces
from gym.spaces.box import Box
from PIL import Image
from matplotlib import pyplot as plt

#################################
#####  Breakout Environment  ######
#################################
class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        '''工夫1のNo-Operationです。リセット後適当なステップの間何もしないようにし、
        ゲーム開始の初期状態を様々にすることｆで、特定の開始状態のみで学習するのを防ぐ'''
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(
                1, self.noop_max + 1)  # pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)

class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        '''工夫2のEpisodic Lifeです。1機失敗したときにリセットし、失敗時の状態から次を始める'''
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        '''5機とも失敗したら、本当にリセット'''
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        '''工夫3のMax and Skipです。4フレーム連続で同じ行動を実施し、最後の3、4フレームの最大値をとった画像をobsにする'''
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros(
            (2,)+env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        obs_buffer = np.copy(self._obs_buffer)  # Copy for ray imutable error: https://github.com/ray-project/ray/issues/369
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                obs_buffer[0] = obs
            if i == self._skip - 1:
                obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env):
        '''工夫4のWarp frameです。画像サイズをNatureのDQN論文と同じ84x84の白黒にします'''
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.height, self.width, 1), dtype=np.uint8)

    def observation(self, frame):
        transform = T.Compose([T.ToPILImage(),T.Grayscale(),
                    T.Resize((self.width, self.height), interpolation=T.InterpolationMode.BICUBIC),
                    T.ToTensor()])
        observation = torch.from_numpy(frame.transpose(2,0,1).astype(np.float32)).clone()
        observation = transform(observation).unsqueeze(0)
        return observation

class WrapPyTorch(gym.ObservationWrapper):
    def __init__(self, env=None):
        '''PyTorchのミニバッチのインデックス順に変更するラッパー'''
        super(WrapPyTorch, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        observation = torch.from_numpy(observation.transpose(2, 0, 1).astype(np.float32)).clone().unsqueeze(0)
        return observation



class EnvParameter(NamedTuple):
    max_lot: int    # 最大数量
    spread: int     # スプレッド
    window_size: int
        
class BreakoutEnv(gym.Wrapper):
    def __init__(self):
        env = gym.make('Breakout-v0').unwrapped
        gym.Wrapper.__init__(self, env)
        self.env = NoopResetEnv(self.env, noop_max=30)
        self.env = MaxAndSkipEnv(self.env, skip=4)
        self.env = EpisodicLifeEnv(self.env)
        self.env = WarpFrame(self.env)
        self.action_space = spaces.Discrete(3) # action is [0, 1, 2] 0: NOOP, 1: RIGHT -> 2, 2: LEFT -> 3
        self.action_list = [0,2,3,1] # index 3 is Fire option : for fire action at first
        print(self.action_space.n)
        #self.env = WrapPyTorch(self.env)
        
    def reset(self):
        observation = self.env.reset() # (210, 160, 3) (h,w,c)

        # fire action at first
        fire_action = 3
        observation, _, _, _ = self.step(fire_action)
        state = observation
        return state
        
    def step(self, action): 
        # action is [0, 1, 2] 0: NOOP, 1: RIGHT -> 2, 2: LEFT -> 3
        action = self.action_list[action]
        observation, reward, done, info = self.env.step(action)
        state = observation
        #print(observation.squeeze(0).shape)
        #plt.imshow(observation.squeeze(0).squeeze(0))
        #plt.show()

        if done:
            state = None

        return state, reward, done, info

#################################
#####  Cartpole Environment  ####
#################################

class CartpoleEnv(gym.Wrapper):
    def __init__(self):
        env = gym.make('CartPole-v0').unwrapped
        gym.Wrapper.__init__(self, env)
        self.episode_step = 0
        self.complete_episodes = 0
        
    def step(self, action): 
        observation, reward, done, info = self.env.step(action)
        self.episode_step += 1

        state = torch.from_numpy(observation).type(torch.FloatTensor)  # numpy変数をPyTorchのテンソルに変換
        state = torch.unsqueeze(state, 0)

        if self.episode_step == 200: # 200以上でdoneにする
            done = True

        if done:
            state = None
            if self.episode_step > 195:
                reward = 1
                self.complete_episodes += 1  # 連続記録を更新
                #if self.complete_episodes >= 10:
                #    print("{}回連続成功".format(self.complete_episodes))
            else:
                # こけたら-1を与える
                reward = -1
                self.complete_episodes = 0
            
            self.episode_step = 0

        return state, reward, done, info

    def reset(self):
        observation = self.env.reset()
        state = torch.from_numpy(observation).type(torch.FloatTensor)  # numpy変数をPyTorchのテンソルに変換
        state = torch.unsqueeze(state, 0)
        return state
