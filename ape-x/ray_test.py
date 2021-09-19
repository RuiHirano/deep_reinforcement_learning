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
def worker_func2(pid):
    time.sleep(5)
    return f"pid {pid} finished"

def use_ray():
    ray.init()
    start = time.time()
    results = [worker_func2.remote(i) for i in range(3)]
    print(ray.get(results)) 
    print("Elapsed:", time.time() - start) 

@ray.remote
def worker_func3(pid):
    """各プロセスの実行には3-15秒かかる
    """
    time.sleep(random.randint(3, 15))
    return f"pid {pid} finished"

def use_ray_wait():
    ray.init()
    start = time.time()
    work_in_progresses = [worker_func3.remote(i) for i in range(10)]

    for i in range(10):
        finished, work_in_progresses = ray.wait(work_in_progresses, num_returns=1)
        orf = finished[0]
        print(len(work_in_progresses))
        print(ray.get(orf))
        print("Elapsed:", time.time() - start)

@ray.remote
class Worker:

    def __init__(self, worker_id):

        self.worker_id = worker_id

        self.n = 0

    def add(self, n):
        print("add")
        time.sleep(5)

        self.n += n

    def get_value(self):

        return f"Process {self.worker_id}: value: {self.n}"

def use_ray_class():
    ray.init(num_cpus=2)
    start = time.time()

    workers = [Worker.remote(i) for i in range(5)]
    for worker in workers:
        worker.add.remote(10)
        
    for worker in workers:
        worker.add.remote(5)

    for worker in workers:
        print(ray.get(worker.get_value.remote()))

    print("Elapsed:", time.time() - start)

@ray.remote
class Worker2:

    def __init__(self):
        pass

    def run(self):
        count = 0
        while True:
            time.sleep(1)
            count+=1
            print(count)


def cancel_test():
    try:
        worker = Worker2.remote()
        run = worker.run.remote()
        time.sleep(5)
    except KeyboardInterrupt:
        print("shutdown")
        ray.shutdown()
    

#no_ray()
#use_ray()
#use_ray_wait()
print(psutil.cpu_count(logical=True))
cancel_test()
#print(psutil.cpu_count())
#use_ray_class()