#!/usr/bin/env python
import argparse
from functools import partial

import gym
from gym.wrappers import Monitor
import tensorflow as tf
import tflearn

import rl.env # noqa : F401
# the above registers the Magnets environment
from rl.dqn import DQNAgent
from rl.hierarchical import HierarchicalAgent
from rl.policy import GreedyPolicy, LinearDecayGreedyEpsilonPolicy

TRAIN_STEPS = 5000000
TEST_STEPS = 100

EPSILON_START = 0.5
EPSILON_END = 0.01
EPSILON_STEPS = 10000

VIDEO_INTERVAL = 10000


def single_agent_model(x, n_actions):
    with tf.name_scope("single_agent"):
        h1 = tflearn.fully_connected(x, 20, activation="relu")
        out = tflearn.fully_connected(h1, n_actions)
        return out


def better_single(x, n_agents, n_actions):
    h1 = tflearn.fully_connected(x, 20, activation="relu")
    h2 = tflearn.fully_connected(h1, 20, activation="relu")
    outs = []
    for i in range(n_agents):
        outi = tflearn.fully_connected(h2, n_actions)
        outs.append(outi)
    return tf.stack(outs, axis=1, name="q_pred")


def sequential_model(x, n_agents, n_actions):
    qs = []
    smaxes = []
    for i in range(n_agents):
        x_i = tf.concat(smaxes + [x], axis=1)
        qs.append(single_agent_model(x_i, n_actions))
        smaxes.append(tflearn.activations.softmax(qs[-1]))
    out = tf.stack(qs, axis=1, name="q_pred")
    return out


def parallel_model(x, n_agents, n_actions):
    qs = []
    for _ in range(n_agents):
        qs.append(single_agent_model(x, n_actions))
    out = tf.stack(qs, axis=1, name="qpred")
    return out


def sequential_message_model(x, n_agents, n_actions):
    """ some notes:
        some choices in network architecture are arbitrary. For example there
        is no good reason for the number of units in the hidden layer to be 20.
        Also, there is no good reason for the size of the message to be 9. We
        should tune these numbers as we go.
    """
    # l1 is some hidden layer output that agent 1 gives
    l1 = tflearn.fully_connected(x, 100, activation="relu")
    l2 = tflearn.fully_connected(l1, 100, activation="relu")
    q1 = tflearn.fully_connected(l2, n_actions)

    # the message passed from agent 1 to agent 2 is h1
    h1 = tflearn.fully_connected(l2, n_actions)

    messages = [h1]
    qs = [q1]

    print("Gets past the first agent")

    for i in range(n_agents - 2):
        # input_i is the message + state input
        print(i, "Processing input")
        input_i = tf.concat([x, messages[i]], axis=1)

        print(i, "done stacking, passing through network")

        # li is some hidden layer for agent i
        li = tflearn.fully_connected(input_i, 100, activation="relu")
        li2 = tflearn.fully_connected(li, 100, activation="relu")
        qi = tflearn.fully_connected(li2, n_actions)

        # message from agent i+2 to agent i+3
        hi = tflearn.fully_connected(li2, 2 * n_actions)

        messages.append(hi)
        qs.append(qi)

    input_n = tf.concat([x, messages[n_agents-2]], axis=1)
    qn = single_agent_model(input_n, n_actions)

    qs.append(qn)

    out = tf.stack(qs, axis=1, name="q_pred")
    return out


def no_video_schedule(i):
    return False


def regular_video_schedule(n, i):
    return i % n == 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--env', default='Magnets-v0', help='Atari env name')
    parser.add_argument('--name', default='dqn', type=str, help='name of model')
    parser.add_argument('--run', default=False, action='store_true', help='evaluate model')
    parser.add_argument('--train', default=False, action='store_true', help='train model')
    parser.add_argument('--monitor', default=False, action='store_true',
                        help='use the gym monitor wrapper to monitor progress')
    parser.add_argument('--video', default=False, action='store_true',
                        help='when using monitor, also store video')
    parser.add_argument('--iterations', default=None, type=int,
                        help='when evaluating, use the model trained for this many iterations')
    parser.add_argument('--model', type=str, default='parallel',
                        choices=['parallel', 'sequential', 'message', 'better'])
    parser.add_argument('--hierarch', default=False, action='store_true', help='hierarchical')

    args = parser.parse_args()

    if (not (args.run or args.train)):
        print("Must specify at least one of --run or --train")
        parser.print_help()
        exit(1)

    if (args.model == 'parallel'):
        q_model = parallel_model
    elif (args.model == 'sequential'):
        q_model = sequential_model
    elif (args.model == 'message'):
        q_model = sequential_message_model
    elif (args.model == 'better'):
        q_model = better_single
    else:
        print("unrecognized model %s" % args.model)
        parser.print_help()
        exit(1)

    if args.video:
        video_callable = partial(regular_video_schedule, VIDEO_INTERVAL)
    else:
        video_callable = no_video_schedule

    env = gym.make(args.env)
    if (args.monitor):
        env = Monitor(env, "%s/monitor" % args.name, video_callable=video_callable)

    if not args.hierarch:
        agent = DQNAgent(env, q_model, model_name=args.name)
    else:
        agent = HierarchicalAgent(env, single_agent_model, 3, model_name=args.name) 
    train_policy = LinearDecayGreedyEpsilonPolicy(EPSILON_START, EPSILON_END, EPSILON_STEPS)
    test_policy = GreedyPolicy()

    if (args.train):
        agent.train(train_policy, TRAIN_STEPS)
        env.close()
    if (args.run):
        (m, v) = agent.evaluate(test_policy, TEST_STEPS, args.iterations)
        print("mean: %f, variance: %f" % (m, v))
