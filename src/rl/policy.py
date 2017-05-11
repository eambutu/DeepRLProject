import numpy as np


class GreedyPolicy:
    def select_action(self, q_values):
        return np.argmax(q_values, axis=-1)


class GreedyEpsilonPolicy:
    def __init__(self, epsilon):
        assert epsilon > 0 and epsilon < 1
        self.epsilon = epsilon

    def select_action(self, q_values, multiple_agent=True):
        n_agents = q_values.shape[0]
        n = q_values.shape[-1]
        x = np.argmax(q_values, axis=-1)
        if (np.random.binomial(1, self.epsilon) == 0):
            return x
        else:
            if multiple_agent:
                return np.random.randint(n, size=n_agents)
            else:
                return np.random.randint(n)


class LinearDecayGreedyEpsilonPolicy:
    def __init__(self, start_value, end_value, num_steps):
        assert start_value > end_value
        self.start_value = start_value
        self.end_value = end_value
        self.num_steps = num_steps
        self.cur_steps = 0

    def select_action(self, q_values, multiple_agent=True):
        eps = self.start_value + \
            min(1.0, float(self.cur_steps)/float(self.num_steps))\
            * (self.end_value - self.start_value)

        n_agents = q_values.shape[0]
        n = q_values.shape[-1]
        x = np.argmax(q_values, axis=-1)
        self.cur_steps += 1

        if (np.random.binomial(1, eps) == 0):
            return x
        else:
            if multiple_agent:
                return np.random.randint(n, size=n_agents)
            else:
                return np.random.randint(n)


class UniformRandomPolicy:
    def __init__(self):
        return

    def select_action(self, q_values, multiple_agent=True):
        n_agents = q_values.shape[0]
        n = q_values.shape[-1]
        if multiple_agent:
            return np.random.randint(n, size=n_agents)
        else:
            return np.random.randint(n)
