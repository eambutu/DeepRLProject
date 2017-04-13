#!/usr/bin/env python
import argparse
from threading import Thread

import gym
import tflearn

import rl.env # noqa : F401
# the above registers the Magnets environment
from rl.dqn import DQNAgent
from rl.policy import GreedyPolicy, LinearDecayGreedyEpsilonPolicy

TRAIN_STEPS = 5000000
TEST_STEPS = 100

EPSILON_START = 0.5
EPSILON_END = 0.01
EPSILON_STEPS = 10000


def q_model(x, n):
    h1 = tflearn.fully_connected(x, 20, activation='relu')
    out = tflearn.fully_connected(h1, n)
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--env', default='MagnetsMulti-v0',
                        help='Environment name. Must be one that supports many agents')
    parser.add_argument('--name', default='dqn', type=str, help='name of model')
    parser.add_argument('--run', default=False, action='store_true', help='evaluate model')
    parser.add_argument('--train', default=False, action='store_true', help='train model')
    parser.add_argument('--iterations', default=None, type=int,
                        help='when evaluating, use the model trained for this many iterations')

    args = parser.parse_args()

    if (not (args.run or args.train)):
        print("Must specify at least one of --run or --train")
        parser.print_help()
        exit(1)

    multienv = gym.make(args.env)
    agents = []
    for (i, env) in enumerate(multienv.child_envs):
        agents.append(DQNAgent(env, q_model, model_name="%s/agent-%d" % (args.name, i)))

    train_policy = LinearDecayGreedyEpsilonPolicy(EPSILON_START, EPSILON_END, EPSILON_STEPS)
    test_policy = GreedyPolicy()

    if (args.train):
        train_args = (train_policy, TRAIN_STEPS)
        train_threads = [Thread(target=agent.train, args=train_args) for agent in agents]
        for thread in train_threads:
            thread.start()
        for thread in train_threads:
            thread.join()
    if (args.run):
        test_args = (test_policy, TEST_STEPS, args.iterations)
        test_threads = [Thread(target=agent.evaluate, args=test_args) for agent in agents]
        for thread in test_threads:
            thread.start()
        for thread in test_threads:
            thread.join()
