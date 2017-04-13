import time
import math
from threading import Lock, get_ident
from collections import Iterable

import numpy as np
from gym import Env
from gym.spaces import Discrete, Box
from gym.envs.registration import register
from PIL import Image, ImageDraw

from rl.state import State
from rl.threshlock import ThreshLock

ENV_UPPER = math.sqrt(2)
ENV_LOWER = -math.sqrt(2)
ENV_SIDE = ENV_UPPER - ENV_LOWER

BOUND_UPPER = 1
BOUND_LOWER = -1
BOUND_SIDE = BOUND_UPPER - BOUND_LOWER

RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

RENDER_WIDTH = 400
RENDER_HEIGHT = 400
RENDER_AGENT_SIZE = 4


class MagnetsEnv(Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, G_const=1.0, acceleration=30.0, time_step=0.01,
                 time_limit=10, friction=10.0, seed=None,
                 boundary_less=-1, boundary_greater=1, num_agents=3):
        ''' constants '''
        self.G_const = G_const
        self.acceleration = acceleration
        self.time_step = time_step
        self.time_limit = time_limit
        self.friction = friction
        if (seed is None):
            self.seed = int(time.time())
        else:
            self.seed = seed
        self.boundary_less = boundary_less
        self.boundary_greater = boundary_greater
        self.num_agents = num_agents

        self.action_space = Discrete(9**num_agents)

        # It's unclear what low and high here should be. Set them to 0 so
        # that if anyone tries to use them, it is more likely that obviously
        # wrong things happen.
        self.observation_space = Box(low=0, high=0, shape=(4*(num_agents+1),))

        ''' variables that change with time '''
        self.state = State(num_agents, seed)

        self.viewer = None

    def _reset(self):
        self.seed += 1
        self.state = State(self.num_agents, self.seed)
        return self.state.to_array()

    def _action_scal2vec(self, action):
        vec_action = np.zeros(self.num_agents)
        for i in range(self.num_agents):
            vec_action[i] = action % 9
            action /= 9
        return vec_action

    def _step(self, action):
        ''' evolve the state  '''
        if not isinstance(action, Iterable):  # if we didn't get a list of actions
            action = self._action_scal2vec(action)

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

        ''' checking if the game has ended so can return '''
        if (not self.state.in_box()):
            return self.state.to_array(), 0, True, {"Msg": "Game over"}

        return self.state.to_array(), 1, False, {"Msg": "Game not over"}

    def print_state(self):
        self.state.print_state()

    def _render_object(self, draw, obj, color):
        obj_x = int(((obj.pos[0] - ENV_LOWER) / ENV_SIDE)*RENDER_WIDTH)
        obj_y = int(((obj.pos[1] - ENV_LOWER) / ENV_SIDE)*RENDER_HEIGHT)
        draw.arc(
            [obj_x - RENDER_AGENT_SIZE/2, obj_y - RENDER_AGENT_SIZE/2,
             obj_x + RENDER_AGENT_SIZE/2, obj_y + RENDER_AGENT_SIZE/2],
            0, 360,
            fill=color
        )

    def _render_objective(self, draw, color):
        BOUND_X = (BOUND_SIDE/ENV_SIDE)*RENDER_WIDTH/2
        BOUND_Y = (BOUND_SIDE/ENV_SIDE)*RENDER_HEIGHT/2
        draw.rectangle(
            [RENDER_WIDTH/2 + BOUND_X, RENDER_HEIGHT/2 + BOUND_Y,
             RENDER_WIDTH/2 - BOUND_X, RENDER_HEIGHT/2 - BOUND_Y],
            outline=color
        )

    def _render_bounds(self, draw, color):
        draw.arc([0, 0, RENDER_WIDTH, RENDER_HEIGHT], 0, 360, fill=color)

    def _render(self, mode='human', close=False):
        img = Image.new('RGB', (RENDER_HEIGHT, RENDER_WIDTH), WHITE)
        draw = ImageDraw.Draw(img)
        for i in range(self.num_agents):
            agent = self.state.agent_states[i]
            self._render_object(draw, agent, RED)
        self._render_object(draw, self.state.target_state, BLUE)
        self._render_objective(draw, GREEN)
        self._render_bounds(draw, BLACK)
        del draw

        if mode == 'human':
            if (self.viewer is None):
                # don't import SimpleImageViewer by default because even importing
                # it requires a display
                from gym.envs.classic_control.rendering import SimpleImageViewer
                self.viewer = SimpleImageViewer()
            self.viewer.imshow(np.asarray(img))
        elif mode == 'rgb_array':
            return np.asarray(img)


class MultiEnv(MagnetsEnv):
    def __init__(self, *args, **kwargs):
        super(MultiEnv, self).__init__(*args, **kwargs)
        self.agent_envs = \
            [MultiEnvAgentEnv(self, i, self.num_agents) for i in range(self.num_agents)]

        self.action = np.zeros(self.num_agents)
        self.result = None
        self.render_tid = None

        self.action_lock = Lock()

        self.tlock = ThreshLock(self.num_agents)
        # TODO: there's so much synchronization going on here. Could this be optimized?

    def _update_action(self, action_u):
        self.action_lock.acquire()
        self.action = np.maximum(self.action, action_u)
        self.action_lock.release()

    def _reset(self):
        n = self.tlock.wait()
        if (n == 1):  # only one agent should /actually/ call reset
            self.result = super(MultiEnv, self)._reset()
        self.tlock.wait()
        return self.result

    def _step(self, action):
        self._update_action(action)
        n = self.tlock.wait()
        if (n == 1):  # only one agent should /actually/ step
            self.result = super(MultiEnv, self)._step(self.action)
            self.action = np.zeros(self.num_agents)
        self.tlock.wait()
        return self.result

    def _render(self, *args, **kwargs):
        n = self.tlock.wait()
        tid = get_ident()
        if (self.render_tid is None and n == 1):
            self.render_tid = tid
        if (tid == self.render_tid):
            super(MultiEnv, self)._render(*args, **kwargs)

    def _seed(self, *args, **kwargs):
        super(MultiEnv, self)._seed(*args, **kwargs)

    @property
    def child_envs(self):
        return self.agent_envs


class MultiEnvAgentEnv(Env):
    def __init__(self, parentenv, i_agent, n_agents):
        self.env = parentenv
        self.i = i_agent
        self.n_agents = n_agents
        self.observation_space = parentenv.observation_space
        self.action_space = \
            Discrete(int(parentenv.action_space.n ** (1/float(n_agents))))
        self.metadata = parentenv.metadata

    def _step(self, action):
        c_action = np.zeros(self.n_agents)
        c_action[self.i] = action
        return self.env._step(c_action)

    def _reset(self):
        return self.env._reset()

    def _render(self, *args, **kwargs):
        return self.env._render()

    def _seed(self, *args, **kwargs):
        return self.env._seed(*args, **kwargs)


register(
    id='Magnets-v0',
    entry_point='rl.env:MagnetsEnv',
)

register(
    id='Magnets1-v0',
    entry_point='rl.env:MagnetsEnv',
    kwargs={'num_agents': 1, 'friction': 2.0}
)

register(
    id='Magnets2-v0',
    entry_point='rl.env:MagnetsEnv',
    kwargs={'num_agents': 2}
)

register(
    id='MagnetsMulti-v0',
    entry_point='rl.env:MultiEnv',
)


def main():
    test_env = MagnetsEnv()
    for i in range(50):
        test_env.print_state()
        test_env.step([1, 2, 1])
        test_env.render()


if __name__ == '__main__':
    main()
