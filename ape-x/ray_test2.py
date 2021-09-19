import time
import ray
import random
import psutil
def worker_func(pid):
    time.sleep(5)
    return f"pid {pid} finished"

def no_ray():
    start = time.time()
    results = [worker_func(i) for i in range(3)]
    print(results) 
    print("Elapsed:", time.time() - start) 

@ray.remote
class Actor():
    def __init__(self, replay, network):
        self.replay = replay
        self.network = network
        self.batch_size = 32
        self.buffer = []
        self.num_rollout = 30000
        self.q_network = 0
        self.test = "test"

    def run(self):
        for i in self.num_rollout:
            transition = i
            self.buffer.append(transition)
            if self.buffer >= self.batch_size:
                transitions = random.sample(self.buffer, self.batch_size) 
                priority = 0
                self.replay.add(priority)
        
    def update_network(self):
        self.q_network = 1
        self.buffer = []

    def add_replay(self):
        while ray.get(self.network.get.remote()) != "update":
            time.sleep(1)
            print(self.test)
            self.replay.add.remote("actor")

        print("finished")

    def update(self):
        print("update actor")
        self.test = "update test"

@ray.remote
class Network():
    def __init__(self):
        self.text = "default"
    def set(self, text):
        self.text = text
    def get(self):
        return self.text


@ray.remote
class Learner():
    def __init__(self, replay, actor, network):
        self.replay = replay
        self.actor = actor
        self.network = network
        self.batch_size = 32
        self.buffer = []
        self.num_update = 30000
        self.num_remove_replay = 30000
        self.q_network = 0
        self.t_network = 0

    def run(self):
        for i in self.num_update:
            transitions = self.replay.sample()
            self.replay.set_priority()
            if i % self.num_remove_replay == 0:
                self.replay.remove_to_fit()
                self.actors.update_network()

    def check_replay(self):
        time.sleep(5)
        get_buffer = self.replay.get.remote()
        buffer = ray.get(get_buffer, timeout=0)
        print("buffer", buffer)
        ray.get(self.network.set.remote("update"))
        print("update from learner")
        
@ray.remote
class Replay():
    def __init__(self):
        self.replay = ""
        self.batch_size = 32
        self.buffer = []
        self.num_update = 30000
        self.num_remove_replay = 30000
        self.q_network = 0
        self.t_network = 0

    def sample(self, batch_size):
        if len(self.buffer) >= batch_size:
            transitions = random.sample(self.buffer, self.batch_size) 
            return transitions
        return None
    
    def add(self, transition):
        self.buffer.append(transition)

    def set_priority(self, id, priority):
        pass

    def get(self):
        return self.buffer


def test():
    replay = Replay.remote()
    network = Network.remote()
    actor = Actor.remote(replay, network)
    learner = Learner.remote(replay, actor, network)
    actor.add_replay.remote()
    ray.get(learner.check_replay.remote())
    time.sleep(4)

#no_ray()
#use_ray()
#print(psutil.cpu_count())
#use_ray_class()
test()