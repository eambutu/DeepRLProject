from threading import Lock
from time import sleep
import argparse

import numpy as np
from pynput import keyboard

from env import MagnetsEnv

key2move = {
    'w': np.array([[0,  1], [0, 0], [0, 0]]),
    'a': np.array([[-1, 0], [0, 0], [0, 0]]),
    's': np.array([[0, -1], [0, 0], [0, 0]]),
    'd': np.array([[1,  0], [0, 0], [0, 0]]),

    'i': np.array([[0, 0], [0,  1], [0, 0]]),
    'j': np.array([[0, 0], [-1, 0], [0, 0]]),
    'k': np.array([[0, 0], [0, -1], [0, 0]]),
    'l': np.array([[0, 0], [1,  0], [0, 0]]),

    keyboard.Key.up:    np.array([[0, 0], [0, 0], [0,  1]]),
    keyboard.Key.left:  np.array([[0, 0], [0, 0], [-1, 0]]),
    keyboard.Key.down:  np.array([[0, 0], [0, 0], [0, -1]]),
    keyboard.Key.right: np.array([[0, 0], [0, 0], [1,  0]]),
}

jslock = Lock()
joystick = np.array([[0, 0], [0, 0], [0, 0]])
is_running = True
should_reset = False

move2action = {
    (1, 0):   0,
    (1, -1):  1,
    (0, -1):  2,
    (-1, -1): 3,
    (-1, 0):  4,
    (-1, 1):  5,
    (0, 1):   6,
    (1, 1):   7,
    (0, 0):   8,
}


def joystick2action(js):
    return list(map(lambda t: move2action[tuple(t)], js))


def process_key(key):
    try:
        return key.char
    except AttributeError:
        return key


def on_press(key):
    global jslock
    global joystick
    global is_running
    global should_reset
    key = process_key(key)
    if (key == 'q' or key == keyboard.Key.esc):
        is_running = False
    if (key == 'r'):
        should_reset = True
    if (key in key2move.keys()):
        jslock.acquire(blocking=True)
        joystick += key2move[key]
        np.clip(joystick, -1, 1, joystick)
        jslock.release()


def on_release(key):
    global joystick
    key = process_key(key)
    if (key in key2move.keys()):
        jslock.acquire(blocking=True)
        joystick -= key2move[key]
        jslock.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--timestep', type=float, default=0.1,
                        help='time to spend in between simulation steps')
    parser.add_argument('--sim_G', type=float, default=None,
                        help='simulation gravitational constant')
    parser.add_argument('--sim_accel', type=float, default=None,
                        help='time to spend in between simulation steps')
    parser.add_argument('--sim_friction', type=float, default=None,
                        help='time to spend in between simulation steps')
    parser.add_argument('--sim_timestep', type=float, default=None,
                        help='amount of time to simulate in a single simulation step')
    args = parser.parse_args()

    env_kwargs = {}
    if (args.sim_G is not None):
        env_kwargs['G'] = args.sim_G
    if (args.sim_accel is not None):
        env_kwargs['acceleration'] = args.sim_accel
    if (args.sim_friction is not None):
        env_kwargs['friction'] = args.sim_friction
    if (args.sim_timestep is not None):
        env_kwargs['time_step'] = args.sim_timestep

    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        env = MagnetsEnv(num_agents=3, **env_kwargs)
        env.render()
        while is_running:
            if (should_reset):
                env.reset()
                should_reset = False
            jslock.acquire(blocking=True)
            action = joystick2action(joystick)
            jslock.release()
            env.step(action)
            env.render()
            sleep(args.timestep)
