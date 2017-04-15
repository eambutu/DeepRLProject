import os

import numpy as np
import tensorflow as tf

from rl.core import ReplayMemory, Sample

# default hyperparameters
GAMMA = 0.9
ALPHA = 0.001
NUM_BURN_IN = 32
TARGET_UPDATE_INTERVAL = 1
TRAIN_INTERVAL = 1
BATCH_SIZE = 32
MEMORY_SIZE = 10000
REPORT_INTERVAL = 1000


def one_hot(a, n):
    x = np.zeros(shape=(a.size, n))
    for i in range(a.size):
        x[i, a] = 1
    return x


def huber_loss(x, max_grad=1.):
    raw_loss = tf.abs(x)
    return tf.where(
      tf.less(x, max_grad),
      tf.multiply(tf.square(raw_loss), 0.5),
      tf.subtract(tf.multiply(raw_loss, max_grad), 0.5*max_grad*max_grad)
    )


def mean_huber_loss(x, max_grad=1.):
    return tf.reduce_mean(huber_loss(x, max_grad), axis=1)


def process_samples(samples, n):
    s = np.array(list(map(lambda s: s.state, samples)))
    ns = np.array(list(map(lambda s: s.next_state, samples)))
    r = np.array(list(map(lambda s: s.reward, samples)))
    a = np.array(list(map(lambda s: one_hot(s.action, n), samples)))
    t = np.array(list(map(lambda s: s.is_terminal, samples)))
    return (s, a, ns, r, t)


class DQNAgent:
    """
    Class implementing Deep Q-Network Reinforcement Learning Agent,
    with experience replay and target fixing
    """
    def __init__(self,
                 env,
                 q_network,
                 memory=ReplayMemory(MEMORY_SIZE),
                 optimizer=tf.train.AdamOptimizer(learning_rate=ALPHA),
                 loss=mean_huber_loss,
                 gamma=GAMMA,
                 num_burn_in=NUM_BURN_IN,
                 target_update_interval=TARGET_UPDATE_INTERVAL,
                 train_interval=TRAIN_INTERVAL,
                 batch_size=BATCH_SIZE,
                 report_interval=REPORT_INTERVAL,
                 model_name='dqn'):
        """
        Initialize a DQN Agent

        Parameters
        ---------
        env: gym.core.Env
            the environment to use; must have a Discrete or MultiDiscrete
            action space, where each agent has the same number of actions
        q_network: (tf.Tensor, int, int) -> tf.Tensor
            a function that generates your Q-network model, given an input
            state tensor, the number of agents and the number of actions/agent
        memory: rl.core.ReplayMemory
            replay memory
        optimizer: tf.train.Optimizer
            optimizer to use when training
        loss: tf.Tensor -> tf.Tensor
            loss function to use when training
        gamma: float
            discount factor to use during training
        num_burn_in: int
            number of experiences to sample into replay memory before training
        target_update_interval: int
            the number of samples to let pass between target network updates
        train_interval: int
            the number of samples to let pass between network training steps
        batch_size: int
            the number of experiences to sample from replay memory during a
            training step
        report_interval: int
            interval at which to generate progress reports during training

        returns: rl.dqn.DQNAgent
            the initialized dqn agent
        """
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
            self.n_agents = env.action_space.size
            self.interact = self._multidiscrete_interact
        except AttributeError:  # Discrete
            self.n_actions = env.action_space.n
            self.n_agents = 1
            self.interact = self._discrete_interact

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        with tf.variable_scope("train_network"):
            self.train_s = tf.placeholder(tf.float32, (None, ) + env.observation_space.shape)
            self.train_a = tf.placeholder(tf.float32, (None, self.n_agents, self.n_actions))
            self.train_y = tf.placeholder(tf.float32, (None, self.n_agents))
            self.train_q = self.q_network(self.train_s, self.n_agents, self.n_actions)
            cost = self.loss(tf.reduce_sum((self.train_q * self.train_a), axis=-1) - self.train_y)
            self.optimize_op = self.optimizer.minimize(cost, global_step=self.global_step)
            train_vars = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope="train_network")
            avg_q = tf.reduce_mean(self.train_q, axis=-1)

        with tf.variable_scope("target_network"):
            self.target_s = tf.placeholder(tf.float32, (None, ) + env.observation_space.shape)
            self.target_q = self.q_network(self.target_s, self.n_agents, self.n_actions)
            target_vars = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope="target_network")
            self.target_update_op = [
                tf.assign(target_vars[i], train_vars[i]) for i in range(len(target_vars))
            ]

        self.metrics = [cost, avg_q]
        self.latest_metrics = [0.0, 0.0]
        self.metric_str = "loss: %f, average q value: %f"

    def _multidiscrete_interact(self, action):
        return self.env.step(action)

    def _discrete_interact(self, action):
        return self.env.step(np.asscalar(action))

    def _calc_q_update(self, ns, r, t):
        nq = self.sess.run(self.target_q, feed_dict={self.target_s: ns})
        return r + (1-t)*self.gamma*nq.max(axis=-1)

    def _select_action(self, s, policy):
        s = np.expand_dims(s, axis=0)
        q = self.sess.run(self.target_q, feed_dict={self.target_s: s})[0]
        return policy.select_action(q)

    def _train_step(self, s, a, y):
        self.sess.run(self.optimize_op, feed_dict={
            self.train_s: s,
            self.train_a: a,
            self.train_y: y,
        })
        return self._global_step()

    def _global_step(self):
        return tf.train.global_step(self.sess, self.global_step)

    def _calc_metrics(self, s, a, y):
        return self.sess.run(self.metrics, feed_dict={
            self.train_s: s,
            self.train_a: a,
            self.train_y: y,
        })

    def _sync_target_network(self):
        self.sess.run(self.target_update_op)

    def train(self, policy, max_updates):
        """
        Train the DQN Agent

        Parameters
        ---------
        env: gym.core.Env
            gym environment to train on
        policy: rl.policy.Policy
            the action selection policy to use
        num_samples: int
            the number of samples to train for

        returns: void
        """

        init_op = tf.global_variables_initializer()

        saver = tf.train.Saver(max_to_keep=None)

        with tf.Session() as sess:
            self.sess = sess
            checkpoint = tf.train.latest_checkpoint(self.save_dir)
            if (checkpoint is not None):
                print ("restoring from checkpoint file %s" % checkpoint)
                saver.restore(sess, checkpoint)
            else:
                sess.run(init_op)

            n_samples = 0
            n_updates = self._global_step()
            n_episodes = 0
            episode_reward = 0  # keep track of this to see if we are learning
            while (n_updates <= max_updates):
                state = self.env.reset()
                is_terminal = False
                while (not is_terminal):
                    action = self._select_action(state, policy)
                    next_state, reward, is_terminal, _ = self.interact(action)
                    episode_reward += reward
                    self.memory.append(Sample(
                        cur_state=state,
                        action=action,
                        next_state=next_state,
                        reward=reward,
                        is_terminal=is_terminal
                    ))

                    if (n_samples > self.num_burn_in):
                        if (n_samples % self.train_interval == 0):
                            batch = self.memory.sample(self.batch_size)
                            (b_s, b_a, b_ns, b_r, b_t) = \
                                process_samples(batch, self.n_actions)
                            b_y = self._calc_q_update(b_ns, b_r, b_t)
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
                    state = next_state
                n_episodes += 1

    def evaluate(self, policy, num_episodes, iterations=None):
        """
        Evaluate the DQN Agent

        Parameters
        ---------
        env: gym.core.Env
            gym environment to evaluate on
        policy: rl.policy.Policy
            the action selection policy to use
        num_episodes: int
            the number of episodes to evaluate for

        returns: (float, float)
            the mean and variance of the agent's rewards/episode
        """
        if (iterations is None):
            checkpoint = tf.train.latest_checkpoint(self.save_dir)
        else:
            checkpoint = "%s/model-%d" % (self.save_dir, iterations)

        saver = tf.train.Saver(max_to_keep=None)
        with tf.Session() as sess:
            self.sess = sess
            saver.restore(sess, checkpoint)

            samples = []
            for i in range(num_episodes):
                is_terminal = False
                state = self.env.reset()
                episode_reward = 0
                while (not is_terminal):
                    action = self._select_action(state, policy)
                    next_state, reward, is_terminal, _ = self.interact(action)
                    episode_reward += reward
                    self.env.render()
                    state = next_state
                samples.append(episode_reward)

        mean = sum(samples)/len(samples)
        variance = sum(map(lambda x: x**2, samples))/len(samples) - (mean**2)
        return (mean, variance)
