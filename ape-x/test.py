
from lib.env import CartpoleEnv, BreakoutEnv
import time
import random
from matplotlib import pyplot as plt
env = BreakoutEnv()

state = env.reset()
for i in range(1000):

    time.sleep(0.1)
    env.render()
    '''print(state.squeeze(0).shape)
    plt.imshow(state.squeeze(0).squeeze(0))
    plt.show()'''
    action = random.choice([0, 1, 2])
    next_state, reward, done, _ = env.step(action)
    print(i, reward, done)
    if done:
        state = env.reset()
    else:
        state = next_state