#!/usr/bin/env python
import gym
import tensorflow as tf
import numpy as np
import os

import rl.env
from rl.dqn import DQNAgent
from rl.policy import GreedyPolicy, LinearDecayGreedyEpsilonPolicy
from rl.core import ReplayMemory, Sample
from rl.utils import mean_squared_loss, process_samples
from rl.ddpg import ActorNetwork, CriticNetwork

# default hyperparameters for DQN
GAMMA = 0.9
ALPHA = 0.001
NUM_BURN_IN = 32
TARGET_UPDATE_INTERVAL = 1
TRAIN_INTERVAL = 1
BATCH_SIZE = 32
MEMORY_SIZE = 10000
REPORT_INTERVAL = 1000

# Number of steps subcontroller takes
NUM_SUB_STEPS = 10

# Hyperparameters for DDPG
ACTOR_LEARNING_RATE = 0.0001
CRITIC_LEARNING_RATE = 0.001
# Soft target update
TAU = 0.01
# Each entry in the goal vectors will be between [-ACTION_BOUND,
# ACTION_BOUND] (do we even want this?)
ACTION_BOUND = 100


class HierarchicalAgent(DQNAgent):
    """
    Implementing hierarchical with DDPG for the metacontroller,
    and DQN with experience replay and target fixing for subcontrollers

    For now, the Q-network takes in
        (pos/vel of target, pos/vel of agent i, goal_x, goal_y)
    and outputs an action (0 to 8)
    """
    def __init__(self,
                 env,
                 q_network,
                 num_agents,
                 memory=ReplayMemory(MEMORY_SIZE),
                 optimizer=tf.train.AdamOptimizer(learning_rate=ALPHA),
                 loss=mean_squared_loss,
                 gamma=GAMMA,
                 num_burn_in=NUM_BURN_IN,
                 target_update_interval=TARGET_UPDATE_INTERVAL,
                 train_interval=TRAIN_INTERVAL,
                 batch_size=BATCH_SIZE,
                 report_interval=REPORT_INTERVAL,
                 model_name='hierarchical'):
        self.memory = memory
        self.env = env
        self.q_network = q_network
        self.optimizer = optimizer
        self.loss = loss
        self.gamma = gamma
        self.num_burn_in = num_burn_in
        self.target_update_interval = target_update_interval
        self.train_interval = train_interval
        self.batch_size = batch_size
        self.report_interval = report_interval
        self.num_agents = num_agents

        self.model_name = model_name
        self.save_dir = './%s' % model_name
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # Tensorflow graph information
        self.sess = None

        try:  # MultiDiscrete
            # as of yet, we don't support agents having different numbers of
            # actions, or actions not numbered from 0 to their max
            assert (np.all(env.action_space.low == 0))
            assert (np.all(env.action_space.high == env.action_space.high[0]))
            self.n_actions = env.action_space.high[0] - env.action_space.low[0]
            self.n_agents = env.action_space.shape
            self.interact = self._multidiscrete_interact
        except AttributeError:  # Discrete
            self.n_actions = env.action_space.n
            self.n_agents = 1
            self.interact = self._discrete_interact

        self.sub_env = gym.make('MagnetsSubController-v0')

        state_dim = env.observation_space.shape[0]
        action_dim = 2 * self.n_agents

        self.actor = ActorNetwork(self.sess, state_dim, action_dim, ACTION_BOUND,
                                  ACTOR_LEARNING_RATE, TAU)
        self.critic = CriticNetwork(self.sess, state_dim, action_dim, CRITIC_LEARNING_RATE, TAU,
                                    self.actor.get_num_trainable_vars())

        self.metacontroller_memory = ReplayMemory(MEMORY_SIZE)

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        # Rewrite these tensorflow placeholders because our dimensions are
        # different now
        with tf.variable_scope("train_network"):
            # (posx, posy, velx, vely, posxtarget, posytarget,
            # velxtarget, velytarget, goalx, goaly)
            self.train_s = tf.placeholder(tf.float32, (None, 10), name="state")

            self.train_a = tf.placeholder(tf.float32, (None, self.n_actions), name="action")

            self.train_y = tf.placeholder(tf.float32, (None, 1), name="q_actual")
            self.train_q = self.q_network(self.train_s, self.n_actions)
            with tf.name_scope("loss_func"):
                cost = self.loss(tf.reduce_sum((self.train_q * self.train_a), axis=-1)-self.train_y)
            self.optimize_op = self.optimizer.minimize(cost, global_step=self.global_step)
            train_vars = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope="train_network")
            avg_q = tf.reduce_mean(self.train_q, name="avg_q_pred")

        with tf.variable_scope("target_network"):
            self.target_s = tf.placeholder(tf.float32, (None, 10), name="state")
            self.target_q = self.q_network(self.target_s, self.n_actions)
            target_vars = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope="target_network")
            self.target_update_op = [
                tf.assign(target_vars[i], train_vars[i]) for i in range(len(target_vars))
            ]

        self.metrics = [cost, avg_q]
        self.latest_metrics = [0.0, 0.0]
        self.metric_str = "loss: %f, average q value: %f"

    def _get_sub_state(self, state, goal, i):
        # Get the state for the ith agent
        print(state[0:4])
        print(state[4*(i+1):4*(i+2)])
        print(goal[2*i:2*(i+1)])
        return np.concatenate([state[0:4], state[4*(i+1):4*(i+2)], goal[2*i:2*(i+1)]], axis=0)

    def _select_action(self, s, goal, policy):
        action = []
        for i in range(self.num_agents):
            partial_info = self._get_sub_state(s, goal, i)
            partial_info = np.reshape(partial_info, (1, 10))
            q = self.sess.run(self.target_q, feed_dict={self.target_s: partial_info})[0]
            print("Q:")
            print(q)
            print("Selected by policy:")
            print(policy.select_action(q, False))
            action.append(policy.select_action(q, False))
        return np.array(action)

    def _calc_q_update(self, ns, r, t):
        nq = self.sess.run(self.target_q, feed_dict={self.target_s: ns})
        return r + (1-t)*self.gamma*nq.max(axis=-1)

    def train(self, policy, num_eps):
        init_op = tf.global_variables_initializer()

        saver = tf.train.Saver(max_to_keep=None)

        with tf.Session() as sess:
            self.sess = sess
            self.actor.sess = self.sess
            self.critic.sess = self.sess
            checkpoint = tf.train.latest_checkpoint(self.save_dir)
            if (checkpoint is not None):
                print ("restoring from checkpoint file %s" % checkpoint)
                saver.restore(sess, checkpoint)
            else:
                sess.run(init_op)
            summarizer = tf.summary.FileWriter(self.save_dir + '/summary', self.sess.graph)
            summarizer.flush()

            n_samples = 0
            n_episodes = 0
            n_total_episodes = 0
            episode_reward = 0  # keep track of this to see if we are learning
            while (n_total_episodes <= num_eps):
                state = self.env.reset()
                is_terminal = False
                while (not is_terminal):
                    goal = self.actor.predict(np.reshape(state, (1, 16)))[0]

                    number_sub_steps = 0
                    total_reward = 0
                    sub_state = self.sub_env._reset(state, goal)

                    while (number_sub_steps < NUM_SUB_STEPS and not is_terminal):
                        sub_action = self._select_action(sub_state, goal, policy)
                        sub_next_state, sub_reward, is_terminal, _ = self.sub_env.step(sub_action)
                        next_state, reward, _, _ = self.env.step(sub_action)
                        total_reward += reward
                        for i in range(self.n_agents):
                            print("Is terminal: ")
                            print(is_terminal)
                            self.memory.append(Sample(
                                cur_state=self._get_sub_state(sub_state, goal, i),
                                action=sub_action[i],
                                next_state=self._get_sub_state(sub_next_state, goal, i),
                                reward=sub_reward[i],
                                is_terminal=is_terminal
                            ))

                        if (n_samples > self.num_burn_in):
                            if (n_samples % self.train_interval == 0):
                                batch = self.memory.sample(self.batch_size)
                                (b_s, b_a, b_ns, b_r, b_t) = \
                                    process_samples(batch, self.n_actions, multiple_agent=False)
                                print(b_t)
                                b_y = self._calc_q_update(b_ns, b_r, b_t)
                                b_y = np.reshape(b_y, (32, 1))
                                n_updates = self._train_step(b_s, b_a, b_y)
                                self.latest_metrics = self._calc_metrics(b_s, b_a, b_y)
                                if (n_updates % self.target_update_interval == 0):
                                    self._sync_target_network()

                            if (n_updates % self.report_interval == 0):
                                avg_reward = episode_reward/n_episodes if n_episodes != 0 else 0
                                print("reward/episode since last report: %f" % avg_reward)
                                print(self.metric_str % tuple(self.latest_metrics))
                                print("%d experiences sampled" % n_samples)
                                print("%d updates performed" % n_updates)
                                print("")
                                saver.save(sess, "%s/model" % self.save_dir, global_step=n_updates)
                                n_episodes = 0
                                episode_reward = 0

                        n_samples += 1
                        sub_state = sub_next_state

                    self.metacontroller_memory.append(Sample(
                        cur_state=state,
                        action=goal,
                        next_state=next_state,
                        reward=reward,
                        is_terminal=is_terminal
                    ))

                    episode_reward += reward

                    if (self.metacontroller_memory.len() > self.num_burn_in):
                        batch = self.metacontroller_memory.sample(self.batch_size)
                        (b_s, b_a, b_ns, b_r, b_t) = process_samples(batch, self.n_actions, False)

                        target_q = self.critic.predict_target(b_ns, self.actor.predict_target(b_ns))

                        y_i = []
                        for k in range(self.batch_size):
                            if b_t[k]:
                                y_i.append(b_r[k])
                            else:
                                y_i.append(b_r[k] + GAMMA * target_q[k])

                        # Update the critic given the targets
                        pred_q_value, _ = self.critic.train(b_s, b_a,
                                                            np.reshape(y_i, (self.batch_size, 1)))

                        # Update the actor policy using the sampled gradient
                        a_outs = self.actor.predict(b_s)
                        grads = self.critic.action_gradients(b_s, a_outs)
                        self.actor.train(b_s, grads[0])

                        # Update target networks
                        self.actor.update_target_network()
                        self.critic.update_target_network()

                print("Reward at episode %d: %d", n_total_episodes, episode_reward)

                episode_reward = 0
                n_episodes += 1
                n_total_episodes += 1
