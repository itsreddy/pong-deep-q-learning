from collections import deque
import math, random
import numpy as np

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # Randomly sampling data with specific batch size from the buffer
        samp = np.array(random.sample(self.buffer, batch_size))
        return np.vstack(samp[:,0]), samp[:,1], samp[:,2], np.vstack(samp[:,3]), samp[:,4]

    def __len__(self):
        return len(self.buffer)