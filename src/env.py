from gym import Env
from gym.envs.registration import register
import numpy as np
import math
from state import State, ObjState

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

    def _step(self, action):
        ''' evolve the state  '''
        self.num_steps += 1
        pos_inc = self.state.target_state.vel * self.time_step
        self.state.target_state.pos += pos_inc
        total_acc = np.zeros(2)

        for i in range(self.num_agents):
            diff_i = self.state.target_state.pos - self.state.agent_states[i].pos
            dist_square = (diff_i[0] * diff_i[0]) + (diff_i[1] * diff_i[1])
            total_acc += (self.G_const / dist_square) *\
                    (diff_i / math.sqrt(dist_square))
            self.state.agent_states[i].pos += (self.state.agent_states[i].vel *\
                                               self.time_step)
            agent_dist = self.state.agent_states[i].pos[0] ** 2 +\
                         self.state.agent_states[i].pos[1] ** 2

            if (agent_dist > 2):
                self.state.agent_states[i].pos /= agent_dist
                self.state.agent_states[i].pos *= 2

        self.state.target_state.vel += (total_acc * self.time_step)

        ''' update velocities of agents based on acceleration '''
        for i in range(self.num_agents):
            ''' acceleration has constant magnitude and one of 8 directions '''
            acc_dir = np.zeros(2)
            if (action[i] != 8):
                acc_dir = np.asarray([math.cos((i * math.pi) / 4),
                    math.sin((i * math.pi) / 4)])
            vel_inc = self.acceleration * acc_dir * self.time_step
            self.state.agent_states[i].vel += vel_inc

            ''' velocity might be greater than max speed: check for that
                and clip velocity to max speed '''
            vel_mag = self.state.agent_states[i].vel[0] ** 2 +\
                      self.state.agent_states[i].vel[1] ** 2
            if (vel_mag > self.speed_limit):
                self.state.agent_states[i].vel /= vel_mag
                self.state.agent_states[i].vel *= self.speed_limit

        ''' checking if the game has ended so can return '''
        if (not self.state.in_box()):
            return self.state, 0, True, {"Msg": "Game over"}

        return self.state, 1, True, {"Msg": "Game not over"}

    def print_state(self):
        self.state.print_state()

    def _render(self, close):
        pass

def main():
    test_env = MagnetsEnv()
    for i in range(50):
        test_env.print_state()
        test_env.step([1,2,1])

if __name__ == '__main__':
    main()
