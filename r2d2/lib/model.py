from abc import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#################################
#####         Net          ######
#################################
class RecurrentLinearNet(nn.Module):
    '''線形入力のDQN'''
    def __init__(self, num_states, num_actions):
        super(RecurrentLinearNet, self).__init__()
        self.num_actions = num_actions
        self.input = nn.Linear(num_states, 128)
        self.lstm = nn.LSTM(128, 128)
        self.output = nn.Linear(128, num_actions)


    def forward(self, x, states, prev_action):
        #x:  (batch_size, state_size)
        x = F.relu(self.input(x)) # (1, 128) (batch, 128)
        x = x.unsqueeze(0) # (1, 1, 128) (length, batch, 128)
        h_n, c_n = None, None
        if states[0] == None or states[1] == None:
            x, (h_n, c_n) = self.lstm(x)
        else:
            #print("debug", x.size(), states[0].size(), states[1].size())
            x, (h_n, c_n) = self.lstm(x, (states[0], states[1]))
        #print(h_n.size(), c_n.size()) # ((D∗num_layers, batch_size, hidden))
        x = F.relu(x).squeeze(0) # (length, batch, 128) -> (batch, 128)
        x = F.relu(self.output(x)) # (batch, action_size)
        return x, (c_n, h_n)

    def select_action(self, state, states, prev_action, epsilon):
        x, (c_n, h_n) = self(state, states, prev_action)
        sample = random.random()
        if sample > epsilon:
            with torch.no_grad():
                action = np.argmax(x.tolist()) 
        else:
            # ε-greedy
            action = random.randrange(self.num_actions)  
        return action, (c_n, h_n)

class RecurrentCNNNet(nn.Module):
    '''二次元画像入力のDQN'''
    def __init__(self, h, w, outputs):
        super(RecurrentCNNNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

class RecurrentDuelingLinearNet(nn.Module):
    '''線形入力でDuelingNetworkを搭載したDQN'''
    def __init__(self, num_states, num_actions):
        super(RecurrentDuelingLinearNet, self).__init__()
        self.num_states = num_states
        self.num_actions = num_actions

        self.fc1 = nn.Linear(self.num_states, 32)
        self.fcV1 = nn.Linear(32, 32)
        self.fcA1 = nn.Linear(32, 32)
        self.fcV2 = nn.Linear(32, 1)
        self.fcA2 = nn.Linear(32, self.num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))

        V = self.fcV2(self.fcV1(x))
        A = self.fcA2(self.fcA1(x))

        averageA = A.mean(1).unsqueeze(1)
        return V.expand(-1, self.num_actions) + (A - averageA.expand(-1, self.num_actions))

class RecurrentDuelingCNNNet(nn.Module):
    '''二次元画像入力のDuelingDQN'''
    def __init__(self, h, w, num_actions):
        super(RecurrentDuelingCNNNet, self).__init__()
        self.num_actions = num_actions
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=2) # (32, 20, 20)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2) # (64, 9, 9)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=5, stride=2) # (64, 7, 7)
        self.bn3 = nn.BatchNorm2d(64)
        self.flatten = nn.Flatten() # (3136)

        self.dence_V1 = nn.Linear(3136, 512)
        self.value = nn.Linear(512, 1)
        self.dence_A1 = nn.Linear(3136, 512)
        self.advantages = nn.Linear(512, self.num_actions)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x))) # (32, 20, 20)
        x = F.relu(self.bn2(self.conv2(x))) # (64, 9, 9)
        x = F.relu(self.bn3(self.conv3(x))) # (64, 7, 7)
        x = self.flatten(x) # (3136)
        V = self.value(F.relu(self.dence_V1(x))) # 最後の出力はreluしない
        A = self.advantages(F.relu(self.dence_A1(x))) # 最後の出力はreluしない
        averageA = A.mean(1).unsqueeze(1)
        return V.expand(-1, self.num_actions) + (A - averageA.expand(-1, self.num_actions))
        