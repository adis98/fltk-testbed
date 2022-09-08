from fltk.samplers import DistributedSamplerWrapper
import random

# dictionary that maps cifar100 target to a coarse ID and another id within that coarse ID
mapper = {4:(0,0), 30:(0,1), 55:(0,2), 72:(0,3), 95:(0,4),
          1:(1,0), 32:(1,1), 67:(1,2), 73:(1,3), 91:(1,4),
          54:(2,0), 62:(2,1), 70:(2,2), 82:(2,3), 92:(2,4),
          9:(3,0), 10:(3,1), 16:(3,2), 28:(3,3), 61:(3,4),
          0:(4,0), 51:(4,1), 53:(4,2), 57:(4,3), 83:(4,4),
          22:(5,0), 39:(5,1), 40:(5,2), 86:(5,3), 87:(5,4),
          5:(6,0), 20:(6,1), 25:(6,2), 84:(6,3), 94:(6,4),
          6:(7,0), 7:(7,1), 14:(7,2), 18:(7,3), 24:(7,4),
          3:(8,0), 42:(8,1), 43:(8,2), 88:(8,3), 97:(8,4),
          12:(9,0), 17:(9,1), 37:(9,2), 68:(9,3), 76:(9,4),
          23:(10,0), 33:(10,1), 49:(10,2), 60:(10,3), 71:(10,4),
          15:(11,0), 19:(11,1), 21:(11,2), 31:(11,3), 38:(11,4),
          34:(12,0), 63:(12,1), 64:(12,2), 66:(12,3), 75:(12,4),
          26:(13,0), 45:(13,1), 77:(13,2), 79:(13,3), 99:(13,4),
          2:(14,0), 11:(14,1), 35:(14,2), 46:(14,3), 98:(14,4),
          27:(15,0), 29:(15,1), 44:(15,2), 78:(15,3), 93:(15,4),
          36:(16,0), 50:(16,1), 65:(16,2), 74:(16,3), 80:(16,4),
          47:(17,0), 52:(17,1), 56:(17,2), 59:(17,3), 96:(17,4),
          8:(18,0), 13:(18,1), 48:(18,2), 58:(18,3), 90:(18,4),
          41:(19,0), 69:(19,1), 81:(19,2), 85:(19,3), 89:(19,4)}


class BalancedSampler(DistributedSamplerWrapper):
    """
    Distributed Sampler implementation that samples uniformly from the available datapoints, assuming all clients
    have an equal distribution over the data (following the original random seed).
    """
    def __init__(self, dataset, num_replicas=None, rank=None, args=(0, 10)):
        seed, num_task = args
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, seed=seed)
        self.ordered_indices = self.order_by_task(self.dataset)
        self.seed = seed
        self.indices = None
        self.taskID = None
        self.num_replicas = num_replicas
        random.seed(self.seed)
        try:
            taskList = random.sample(self.ordered_indices.keys(), num_task)
            self.taskList = taskList[-rank:] + taskList[:-rank] # rotates the schedule according to the rank
        except ValueError:
            print("More tasks to sample than the task list size!")
            exit(0)
        if rank > 0:
            indices = self.ordered_indices[0]
            random.shuffle(indices)
            self.indices = indices[rank-1:len(indices):self.num_replicas]

    # Only works for CIFAR100!!
    def order_by_task(self, dataset):
        tasks = {}
        for i in range(20):
            tasks[i] = []
        for index, target in enumerate(dataset.targets):
            coarse_id = mapper[target][0]
            tasks[coarse_id].append(index)

        return tasks

    def set_task(self, taskID):
        self.taskID = taskID
        taskIndex = self.taskList[self.taskID]
        indices = self.ordered_indices[taskIndex]
        client_id = self.rank - 1
        random.seed(self.seed)
        random.shuffle(indices) # shuffle the indices in an identical way for every client
        self.indices = indices[client_id:len(indices):self.num_replicas]


