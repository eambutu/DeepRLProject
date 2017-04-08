from gym import Env
# from gym.envs.registration import register
# use the above later

from PIL import Image, ImageDraw
import numpy as np
import time
import math

from state import State

RED = (255, 0, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

RENDER_WIDTH = 400
RENDER_HEIGHT = 400

RENDER_AGENT_SIZE = 20


class MagnetsEnv(Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, G_const=1.0, acceleration=30.0, time_step=0.01,
                 time_limit=10, speed_limit=100.0, friction=10.0, seed=None,
                 boundary_less=-1, boundary_greater=1, num_agents=3):
        ''' constants '''
        self.G_const = G_const
        self.acceleration = acceleration
        self.time_step = time_step
        self.time_limit = time_limit
        self.speed_limit = speed_limit
        self.friction = friction
        if (seed is None):
            self.seed = int(time.time())
        else:
            self.seed = seed
        self.boundary_less = boundary_less
        self.boundary_greater = boundary_greater
        self.num_agents = num_agents

        ''' variables that change with time '''
        self.num_steps = 0
        self.state = State(num_agents, seed)

        self.viewer = None

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
            self.state.agent_states[i].pos += (self.state.agent_states[i].vel *
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
                acc_dir = np.asarray([math.cos((action[i] * math.pi) / 4),
                                     math.sin((action[i] * math.pi) / 4)])
            vel_inc = self.acceleration * acc_dir * self.time_step
            vel_dec = self.friction * self.state.agent_states[i].vel *\
                self.time_step
            self.state.agent_states[i].vel += (vel_inc - vel_dec)

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

        return self.state.state_to_array(), 1, False, {"Msg": "Game not over"}

    def print_state(self):
        self.state.print_state()

    def _render_object(self, draw, obj, color):
        env_side = 2*math.sqrt(2)
        env_lower = -math.sqrt(2)
        obj_x = int(((obj.pos[0] - env_lower) / env_side)*RENDER_WIDTH)
        obj_y = int(((obj.pos[1] - env_lower) / env_side)*RENDER_HEIGHT)
        draw.arc(
            [obj_x - RENDER_AGENT_SIZE/2, obj_y - RENDER_AGENT_SIZE/2,
             obj_x + RENDER_AGENT_SIZE/2, obj_y + RENDER_AGENT_SIZE/2],
            0, 359,
            fill=color
        )

    def _render(self, mode='human', close=False):
        if (self.viewer is None):
            # don't import SimpleImageViewer by default because even importing
            # it requires a display
            from gym.envs.classic_control.rendering import SimpleImageViewer
            self.viewer = SimpleImageViewer()

        img = Image.new('RGB', (RENDER_HEIGHT, RENDER_WIDTH), WHITE)
        draw = ImageDraw.Draw(img)
        for i in range(self.num_agents):
            agent = self.state.agent_states[i]
            self._render_object(draw, agent, RED)
        self._render_object(draw, self.state.target_state, BLUE)
        del draw

        self.viewer.imshow(np.asarray(img))


def main():
    test_env = MagnetsEnv()
    for i in range(50):
        test_env.print_state()
        test_env.step([1, 2, 1])
        test_env.render()


if __name__ == '__main__':
    main()
