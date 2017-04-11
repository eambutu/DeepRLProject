import numpy as np


class GreedyPolicy:
    def select_action(self, q_values):
        return np.argmax(q_values)


class GreedyEpsilonPolicy:
    def __init__(self, epsilon):
        assert epsilon > 0 and epsilon < 1
        self.epsilon = epsilon

    def select_action(self, q_values):
        n = len(q_values)
        x = np.argmax(q_values)
        if (np.random.binomial(1, self.epsilon) == 0):
            return x
        else:
            return np.random.randint(n)


class LinearDecayGreedyEpsilonPolicy:
    def __init__(self, start_value, end_value, num_steps):
        assert start_value > end_value
        self.start_value = start_value
        self.end_value = end_value
        self.num_steps = num_steps
        self.cur_steps = 0

    def select_action(self, q_values, is_training=True):
        eps = self.start_value + \
            min(1.0, float(self.cur_steps)/float(self.num_steps))\
            * (self.end_value - self.start_value)

        n = len(q_values)
        x = np.argmax(q_values)

        if is_training:
            self.cur_steps += 1

            if (np.random.binomial(1, eps) == 0):
                return x
            else:
                return np.random.randint(n)
        else:
            if (np.random.binomial(1, self.end_value) == 0):
                return x
            else:
                return np.random.randint(n)
