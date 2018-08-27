"""

import numpy as np
import random
#experience replay

max_buffer_size = 3
num_experience = 7

#after each rollout, save one buffer
class replayBuffer():
    def __init__(self, buffer_size = 3*max_buffer_size):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience) + len(self.buffer)) - self.buffer_size] = []
        self.buffer.extend(experience)


    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience) + len(self.buffer)) - self.buffer_size] = []
        self.buffer.extend(experience)
    """



def add_sample(self, observation, action, reward, terminal, next_observation, **kwargs):
    if len(self._observations) + len(observation) >= self._max_buffer_size:
        self._observations[0:(len(observation) + len(self._observations)) - self._max_buffer_size] = []
    self._observations.extend(observation)

    if len(self._actions) + len(action) >= self._max_buffer_size:
        self._actions[0:(len(action) + len(self._actions)) - self._max_buffer_size] = []
    self._actions.extend(action)

    if len(self._rewards) + len(reward) >= self._max_buffer_size:
        self._rewards[0:(len(reward) + len(self._rewards)) - self._max_buffer_size] = []
    self._rewards.extend(reward)

    if len(self._terminals) + len(terminal) >= self._max_buffer_size:
        self._terminals[0:(len(terminal) + len(self._terminals)) - self._max_buffer_size] = []
    self._terminals.extend(terminal)

    if len(self._next_obs) + len(next_observation) >= self._max_buffer_size:
        self._next_obs[0:(len(next_observation) + len(self._next_obs)) - self._max_buffer_size] = []
    self._next_obs.extend(next_observation)

    self._advance()

def _advance(self):
    self._top = (self._top + 1) % self._max_buffer_size










    """



    def sample(self, batchsize):
        return np.reshape(np.array(random.sample(self.buffer_size)), [batchsize, 5])

exp_1 = np.array([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3]])
exp1 = np.array([0, 0, 0, 0, 0])
exp2 = np.array([1, 1, 1, 1, 1])
exp3 = np.array([2, 2, 2, 2, 2])
exp4 = np.array([3, 3, 3, 3, 3])

exp_11 = np.array([[10, 10, 10, 10, 10], [11, 11, 11, 11, 11], [12, 12, 12, 12, 12], [13, 13, 13, 13, 13]])
exp11 = np.array([10, 10, 10, 10, 10])
exp21 = np.array([11, 11, 11, 11, 11])
exp31 = np.array([12, 12, 12, 12, 12])
exp41 = np.array([13, 13, 13, 13, 13])


exp111 = {'dic_1' : [100, 100, 100], 'dic_2' : [200, 200, 200, 200]}
exp222 = {'dic_1' : [300, 300, 300], 'dic_2' : [400, 400, 400, 400]}
exp333 = {'dic_1' : [500, 500, 500], 'dic_2' : [600, 600, 600, 600]}
exp444 = {'dic_1' : [700, 700, 700], 'dic_2' : [800, 800, 800, 800]}
exp_111 = {'exp_111': exp111, 'exp_222': exp222, 'exp_333': exp333, 'exp_444': exp444}

rb = replayBuffer()



dexp1 = dict(dic1 = exp1, dic11 = exp11, dic111 = exp111)
dexp2 = dict(dic2 = exp2, dic12 = exp11, dic222 = exp222)
dexp3 = dict(dic3 = exp3, dic13 = exp11, dic333 = exp333)
dexp4 = dict(dic4 = exp4, dic14 = exp11, dic444 = exp444)

print(len(dexp1))
rb.add(dexp1)
print(rb.buffer)
rb.add(dexp2)
print(rb.buffer)
rb.add(dexp3)
print(rb.buffer)
rb.add(dexp4)
print(rb.buffer)


#with update frequency (e.g. 5), save trace of buffers
class experience_buffer():
    def __init__(self, buffer_size = 1000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + 1 >= self.buffer_size:
            self.buffer[0: (1 + len(self.buffer)) - self.buffer_size] = []
        self.buffer.append(experience)

    def sample(self, batch_size, trace_length):
        sampled_episodes = random.sample.(self.buffer, batch_size)
        sampledTraces = []
        for episode in sampled_episodes:
            point = np.random.randint(0, len(episode) + 1 - trace_length)
            sampledTraces.append(episode[point:point + trace_length])
        sampledTraces = np.array(sampledTraces)
        return np.reshape(sampledTraces, [batch_size * trace_length, 5])
    """



