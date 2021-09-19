from abc import *
import torch
import torch.nn as nn
import torch.nn.functional as F

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#################################
#####         Net          ######
#################################
class LinearDQNSample(nn.Module):
    '''線形入力のDQN'''
    def __init__(self, num_states, num_actions):
        super(LinearDQNSample, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(num_states, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, num_actions),
            #nn.ReLU()
        )

    def forward(self, x):
        x = x.to(device)
        logits = self.linear_relu_stack(x)
        return logits

class DQN2DSample(nn.Module):
    '''二次元画像入力のDQN'''
    def __init__(self, h, w, outputs):
        super(DQN2DSample, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
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

class DuelingLinearDQNSample(nn.Module):
    '''線形入力でDuelingNetworkを搭載したDQN'''
    def __init__(self, num_states, num_actions):
        super(DuelingLinearDQNSample, self).__init__()
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
