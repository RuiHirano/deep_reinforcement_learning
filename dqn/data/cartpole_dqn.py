import torch
import torch.nn as nn
import gym

#################################
#####         Net          ######
#################################

class DuelingDQN(nn.Module):
    '''線形入力でDualingNetworkを搭載したDQN'''
    def __init__(self, num_states, num_actions):
        super(DuelingDQN, self).__init__()
        self.num_states = num_states
        self.num_actions = num_actions

        self.fc1 = nn.Linear(self.num_states, 32)
        self.relu = nn.ReLU()
        self.fcV1 = nn.Linear(32, 32)
        self.fcA1 = nn.Linear(32, 32)
        self.fcV2 = nn.Linear(32, 1)
        self.fcA2 = nn.Linear(32, self.num_actions)

    def forward(self, x):
        x = self.relu(self.fc1(x))

        V = self.fcV2(self.fcV1(x))
        A = self.fcA2(self.fcA1(x))

        averageA = A.mean(1).unsqueeze(1)
        return V.expand(-1, self.num_actions) + (A - averageA.expand(-1, self.num_actions))

#################################
#####      Environment     ######
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
                if self.complete_episodes >= 10:
                    print("{}回連続成功".format(self.complete_episodes))
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

def get_env_net():
    env = CartpoleEnv()
    num_actions = env.action_space.n
    num_states = env.observation_space.shape[0]
    net = DuelingDQN(num_states, num_actions)
    return env, net