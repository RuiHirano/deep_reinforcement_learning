
from lib.env import CartpoleEnv, BreakoutEnv
import time
import random
env = BreakoutEnv()

state = env.reset()
for i in range(1000):

    time.sleep(0.1)
    env.render()
    action = random.choice([0, 1, 2])
    next_state, reward, done, _ = env.step(action)
    print(i, reward, done, next_state)
    if done:
        state = env.reset()