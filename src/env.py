from gym import Env
from gym.envs.registration
import numpy as np


class MagnetsEnv(Env):
    def __init__(self, G_const=1.0, acceleration=1.0, time_step=0.01,
                time_limit=10, speed_limit=1.0, seed=29, boundary_less = -1,
                boundary_greater = 1, num_agents = 3):
        ''' constants '''
        self.G_const = G_const
        self.acceleration = acceleration
        self.time_step = time_step
        self.time_limit = time_limit
        self.speed_limit = speed_limit
        self.seed = seed
        self.boundary_less = boundary_less
        self.boundary_greater = boundary_greater
        self.num_agents = num_agents

        ''' variables that change with time '''
        self.num_steps = 0
        self.state = State(num_agents, seed)

    def _reset(self):
        self.__init__()

    def _step(self):
        ''' evolve the state  '''
        self.state.target_pos += (self.state.target_vel * time_step)
        self.num_steps += 1
        total_acc = np.zeros(2)

        for i in range(self.num_agents):
            diff_i = self.state.target_pos - self.state.agent_states[i].pos
            dist_square = (diff[0] * diff[0]) + (diff[1] * diff[1])
            total_acc += (self.G_const / dist_square) * (diff_i / sqrt(dist_square))

        self.state.target_vel += (total_acc * self.time_step)

        ''' update velocities based on acceleration '''


    def _render(self):
        pass
