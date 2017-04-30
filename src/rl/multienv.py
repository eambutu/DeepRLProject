from threading import Lock, get_ident

import numpy as np
from gym import Env
from gym.spaces import Discrete

from rl.threshlock import ThreshLock


class MultiEnv(Env):
    def __init__(self, env, num_agents, *args, **kwargs):
        self.env = env
        self.num_agents = num_agents
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.metadata = env.metadata

        self.action = np.zeros(self.num_agents)
        self.result = None
        self.render_tid = None

        self.action_lock = Lock()

        self.tlock = ThreshLock(self.num_agents)
        # TODO: there's so much synchronization going on here. Could this be optimized?

        self.agent_envs = \
            [MultiEnvAgentEnv(self, i, self.num_agents) for i in range(self.num_agents)]

    def _update_action(self, action_u):
        self.action_lock.acquire()
        self.action = np.maximum(self.action, action_u)
        self.action_lock.release()

    def _reset(self):
        n = self.tlock.wait()
        if (n == 1):  # only one agent should /actually/ call reset
            self.result = self.env._reset()
        self.tlock.wait()
        return self.result

    def _step(self, action):
        self._update_action(action)
        n = self.tlock.wait()
        if (n == 1):  # only one agent should /actually/ step
            self.result = self.env._step(self.action)
            self.action = np.zeros(self.num_agents)
        self.tlock.wait()
        return self.result

    def _render(self, *args, **kwargs):
        n = self.tlock.wait()
        tid = get_ident()
        if (self.render_tid is None and n == 1):
            self.render_tid = tid
        if (tid == self.render_tid):
            # only one agent should /actually/ render
            # also has to be the same thread every time, for some sdl reason
            self.env._render(*args, **kwargs)

    def _seed(self, *args, **kwargs):
        self.env._seed(*args, **kwargs)

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
