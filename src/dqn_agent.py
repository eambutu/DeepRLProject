import argparse

import gym
import tflearn

import rl.env # noqa : F401
# the above registers the Magnets environment
from rl.dqn import DQNAgent
from rl.policy import GreedyPolicy, LinearDecayGreedyEpsilonPolicy

TRAIN_STEPS = 5000000
TEST_STEPS = 20

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_STEPS = 1000000


def q_model(x, n):
    return tflearn.fully_connected(x, n, activation='relu')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--env', default='Magnets-v0', help='Atari env name')
    parser.add_argument('--name', default='dqn', type=str, help='name of model')
    parser.add_argument('--run', default=False, action='store_true', help='evaluate model')
    parser.add_argument('--train', default=False, action='store_true', help='train model')

    args = parser.parse_args()

    if (not (args.run or args.train)):
        print("Must specify at least one of --run or --train")
        parser.print_help()
        exit(1)

    env = gym.make(args.env)
    agent = DQNAgent(env, q_model, model_name=args.name)
    train_policy = LinearDecayGreedyEpsilonPolicy(EPSILON_START, EPSILON_END, EPSILON_STEPS)
    test_policy = GreedyPolicy()

    if (args.train):
        agent.train(train_policy, TRAIN_STEPS)
    if (args.run):
        agent.evaluate(test_policy, TEST_STEPS)
