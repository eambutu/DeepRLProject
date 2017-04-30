import numpy as np


class Sample:
    """Stores a tuple of (s, a, r, s', terminal).

    Parameters
    ---------
    state: numpy float64 array
       in format [target_x, target_y, target_vx, target_vy, agent1_x, ...]
    action: int array
    reward: int
    next_state: numpy float64 array of format above
    is_terminal: boolean
    """
    def __init__(self, cur_state, action, reward, next_state, is_terminal):
        self.state = cur_state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.is_terminal = is_terminal


class ReplayMemory:
    """Interface for replay memory.

    Methods
    ------
    __init__(max_size)
      Initializes replay memory with maximum size
    append(sample)
      Appends a sample (instance of Sample) onto replay memory
    sample(batch_size)
      Returns a randomly chosen batch of size batch_size from replay memory
    """
    def __init__(self, max_size):
        self.index = 0
        self.cur_states = []
        self.next_states = []
        self.rewards = []
        self.actions = []
        self.terminal = []
        self.max_size = max_size

    def append(self, sample):
        if (len(self.cur_states) < self.max_size):
            self.cur_states.append(sample.state)
            self.next_states.append(sample.next_state)
            self.rewards.append(sample.reward)
            self.actions.append(sample.action)
            self.terminal.append(sample.is_terminal)
        else:
            self.cur_states[self.index] = sample.state
            self.next_states[self.index] = sample.next_state
            self.rewards[self.index] = sample.reward
            self.actions[self.index] = sample.action
            self.terminal[self.index] = sample.is_terminal
        self.index = (self.index + 1) % self.max_size

    def sample(self, batch_size):
        samples = []
        sample_idxs = np.random.randint(0, len(self.cur_states), batch_size)

        for i in range(batch_size):
            idx = sample_idxs[i]
            sample = Sample(self.cur_states[idx], self.actions[idx],
                            self.rewards[idx], self.next_states[idx],
                            self.terminal[idx])
            samples.append(sample)

        return samples

    def len(self):
        return len(self.cur_states)
