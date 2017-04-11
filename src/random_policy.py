from random import randint
from time import sleep

from gym.wrappers import Monitor

from env import MagnetsEnv

if __name__ == '__main__':
    env = Monitor(MagnetsEnv(num_agents=3), '/tmp/magnets-rl')
    env.render()
    while(True):
        is_terminal = False
        env.reset()
        while (not is_terminal):
            action = [randint(0, 8), randint(0, 8), randint(0, 8)]
            _, _, is_terminal, _ = env.step(action)
            env.render()
            sleep(0.01)
