#!/usr/bin/env python
import argparse

import gym
from gym.wrappers import Monitor
import tflearn
import tensorflow as tf

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


# our second baseline
# n here is the number of actions for a single agent
def q_model2(x, n=9, num_agents=3):
    input_size = x.get_shape()[0]
    offset = input_size / num_agents
    outputs = []
    for i in range(num_agents):
        xi = tf.slice(x, [i * offset], [offset])
        hi = tflearn.fully_connected(xi, 20, activation='relu')
        out_i = tflearn.fully_connected(hi, n)
        outputs.append(out_i)
    return tf.concat(outputs, 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--env', default='Magnets-v0', help='Atari env name')
    parser.add_argument('--name', default='dqn', type=str, help='name of model')
    parser.add_argument('--run', default=False, action='store_true', help='evaluate model')
    parser.add_argument('--train', default=False, action='store_true', help='train model')
    parser.add_argument('--monitor', default=False, action='store_true',
                        help='use the gym monitor wrapper to monitor progress')
    parser.add_argument('--iterations', default=None, type=int,
                        help='when evaluating, use the model trained for this many iterations')

    args = parser.parse_args()

    if (not (args.run or args.train)):
        print("Must specify at least one of --run or --train")
        parser.print_help()
        exit(1)

    env = gym.make(args.env)
    if (args.monitor):
        env = Monitor(env, "%s/monitor" % args.name)

    agent = DQNAgent(env, q_model, model_name=args.name)
    train_policy = LinearDecayGreedyEpsilonPolicy(EPSILON_START, EPSILON_END, EPSILON_STEPS)
    test_policy = GreedyPolicy()

    if (args.train):
        agent.train(train_policy, TRAIN_STEPS)
    if (args.run):
        (m, v) = agent.evaluate(test_policy, TEST_STEPS, args.iterations)
        print("mean: %f, variance: %f" % (m, v))
